"""Trainer module."""
import argparse
import gc
from contextlib import contextmanager
import dataclasses
from dataclasses import is_dataclass
from distutils.version import LooseVersion
import logging
from pathlib import Path
import time
from typing import Dict, Iterator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import (
    average_nbest_models,
    save_model,
    update_best,
    remove_nbest,
    update_nbest,
)
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochStepScheduler
from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochStepScheduler
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.asr.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter
from espnet2.utils.build_dataclass import build_dataclass

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

from espnet2.utils.import_utils import autocast, GradScaler

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    valid_log_interval: Optional[int]
    no_forward_run: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    history_context_length: int
    complemented_version: int
    concat_decoder: bool

    


class Trainer:
    """Trainer having a optimizer.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    step: int = 0

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        pass

    @staticmethod
    def resume(
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        ngpu: int = 0,
    ):
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states["model"])
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.warning(f"RESUME: The training was resumed using {checkpoint}")

    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options,
        distributed_option: DistributedOption,
        dry_run: list = [],
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()
        if trainer_options.use_amp:
            if LooseVersion(torch.__version__) < LooseVersion("1.6.0"):
                raise RuntimeError("Require torch>=1.6.0 for  Automatic Mixed Precision")
            if trainer_options.sharded_ddp:
                if fairscale is None:
                    raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
                scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
            else:
                scaler = GradScaler()
        else:
            scaler = None

        if trainer_options.resume and (output_dir / "checkpoint.pth").exists():
            cls.resume(
                checkpoint=output_dir / "checkpoint.pth",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(f"The training has already reached at max_epoch: {start_epoch}")

        if distributed_option.distributed:
            if trainer_options.sharded_ddp:
                dp_model = fairscale.nn.data_parallel.ShardedDataParallel(
                    module=model,
                    sharded_optimizer=optimizers,
                )
            else:
                dp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=(
                        # Perform multi-Process with multi-GPUs
                        [torch.cuda.current_device()]
                        if distributed_option.ngpu == 1
                        # Perform single-Process with multi-GPUs
                        else None
                    ),
                    output_device=(torch.cuda.current_device() if distributed_option.ngpu == 1 else None),
                    find_unused_parameters=True,
                    #find_unused_parameters=trainer_options.unused_parameters,
                )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(str(output_dir / "tensorboard"))
        else:
            summary_writer = None

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.warning(f"First Epoch {iepoch}: {iepoch}/{trainer_options.max_epoch}")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            if "TRAIN" not in dry_run:
                train_iterator = cls.build_iterator(train_iter_factory.build_iter(iepoch))
                if trainer_options.log_interval is None:
                    try:
                        trainer_options.log_interval = max(len(train_iterator) // 20, 10)
                    except TypeError:
                        trainer_options.log_interval = 100
                # valid_iterator = cls.build_iterator(valid_iter_factory.build_iter(iepoch))
                # default valid log interval: 1000 steps
                if trainer_options.valid_log_interval is None:
                    trainer_options.valid_log_interval = 1000

                with reporter.observe("train") as sub_reporter:
                    all_steps_are_invalid = cls.train_one_epoch(
                        model=dp_model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        iterator=train_iterator,
                        reporter=sub_reporter,
                        scaler=scaler,
                        summary_writer=summary_writer,
                        options=trainer_options,
                        distributed_option=distributed_option,
                        valid_iterator=valid_iter_factory,
                        iepoch=iepoch,
                        id2order=train_iter_factory.id2order,
                        #valid_iterator=cls.build_iterator(valid_iter_factory.build_iter(iepoch)),
                        
                    )
            else:
                all_steps_are_invalid = False

            # if "PLOT" not in dry_run:
            #     if not distributed_option.distributed or distributed_option.dist_rank == 0:
            #         # att_plot doesn't support distributed
            #         if plot_attention_iter_factory is not None:
            #             with reporter.observe("att_plot") as sub_reporter:
            #                 cls.plot_attention(
            #                     model=model,
            #                     output_dir=output_dir / "att_ws",
            #                     summary_writer=summary_writer,
            #                     iterator=plot_attention_iter_factory.build_iter(iepoch),
            #                     reporter=sub_reporter,
            #                     options=trainer_options,
            #                 )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(reporter.get_value(*trainer_options.val_scheduler_criterion))
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()
            if trainer_options.sharded_ddp:
                for optimizer in optimizers:
                    if isinstance(optimizer, fairscale.optim.oss.OSS):
                        optimizer.consolidate_state_dict()

            # 3-6 save checkpoints
            if "SAVE" not in dry_run and (
                not distributed_option.distributed or distributed_option.dist_rank == 0
            ):
                # 3. Report the results
                logging.info(reporter.log_message())
                if plot_attention_iter_factory is not None:
                    reporter.matplotlib_plot(output_dir / "images")
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer)

                # 4-5. Save/Update the checkpoint
                save_model(model, optimizers, schedulers, scaler, reporter, iepoch, output_dir)

                # 6.1 create sym link if it is the best
                best_epoch, _improved = update_best(trainer_options, reporter, iepoch, output_dir)
                """
                if len(_improved) == 0:
                    logging.warning("There are no improvements in this epoch")
                else:
                    logging.info("The best model has been updated: " + ", ".join(_improved))
                """

                # 6.2 Remove the model files excluding n-best epoch and latest epoch
                """
                _removed = remove_nbest(trainer_options, reporter, iepoch, output_dir, keep_nbest_models)
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))
                """
                update_nbest(trainer_options, reporter, iepoch, output_dir, keep_nbest_models)

                # 6.3 Generated n-best averaged model
                if (
                    trainer_options.nbest_averaging_interval > 0
                    and iepoch % trainer_options.nbest_averaging_interval == 0
                ):
                    logging.info(
                        f"Average nbest model:{keep_nbest_models} by {trainer_options.best_model_criterion}"
                    )
                    average_nbest_models(
                        reporter=reporter,
                        output_dir=output_dir,
                        best_model_criterion=trainer_options.best_model_criterion,
                        nbest=keep_nbest_models,
                    )

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                    trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(f"The training was finished at {trainer_options.max_epoch} epochs ")

    @classmethod
    def build_iterator(cls, this_iterator) -> Iterator:
        class OOMIterator(Iterator):
            def __init__(self, iterator) -> None:
                self.iter = iter(iterator)
                self.oom = False
                self.batch = None

            def shave_batch(self, batch: dict, new_mbsize: int = 1) -> dict:
                old_mbsize = batch["text"].size(0)
                # if it already reduce to 1, skip this mini-batch
                if old_mbsize == 1:
                    new_mbsize = 0
                elif new_mbsize == 0:
                    new_mbsize = old_mbsize // 2
                else:
                    assert new_mbsize > 0

                for k in batch.keys():
                    batch[k] = batch[k][:new_mbsize]
                logging.warning(f"shaveing batch size from {old_mbsize} to {new_mbsize}...")
                return batch

            def __next__(self) -> dict:
                if not self.oom:
                    batch = next(self.iter)[-1]
                else:
                    assert self.batch is not None
                    batch = self.shave_batch(self.batch)
                    self.oom = False
                    if batch["text"].size(0) == 0:
                        batch = next(self.iter)[-1]
                self.batch = batch
                
                return batch
            
            def __iter__(self):
                return self

        return OOMIterator(this_iterator)



    @classmethod
    def update_memory(
        cls,
        mems: torch.Tensor,
        current_encoder_output: torch.Tensor, # [layer_num, batchsize, length, d_head]
        history_context_length: int,
        end_point: list,
        current_length: int,
        index: list,
        masks: torch.Tensor, # [layer_num, 1, length]
        prev_masks: torch.Tensor, 
    ) -> torch.Tensor:
        if history_context_length == 0:
            return tuple(), current_length, index, masks
        if current_length == 0:
            mems = current_encoder_output
            masks = prev_masks
            current_length += 1
            index.append(mems.size(2))
        else:
            if current_length < history_context_length:
                mems = torch.concat((mems, current_encoder_output), 2)
                masks = torch.concat((masks, prev_masks), 2)
                current_length += 1
                index.append(current_encoder_output.size(2))
            else:
                idx = index[0]
                mems = torch.concat((mems, current_encoder_output), 2)
                masks = torch.concat((masks, prev_masks), 2)
                mems = mems[:, :, idx:, :]
                masks = masks[:, :, idx:]
                index = index[1:]
                index.append(current_encoder_output.size(2))

        tmp_mem = mems.clone()
        for i in range(len(end_point)):
            if end_point[i] == 1:
                # tmp_mem = mems.clone()
                tmp_mem[:, i, :, :] = -20
        
        return tmp_mem, current_length, index, masks
    

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        valid_iterator: AbsIterFactory,
        iepoch: int,
        id2order,
        #valid_iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        valid_log_interval = options.valid_log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        # print the consuming time in each part or not, we normally don't need it
        verbose = False

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        iiter = 1
        epoch_start_time = time.perf_counter()
        epoch_acc_num_feats = 0

        mems = tuple()
        history_context_length = options.history_context_length
        complemented_version = options.complemented_version
        concat_decoder = options.concat_decoder
        masks = tuple()
        pos = tuple()
        total_id_list = iterator.iter._index_sampler
        current_length = 0
        index = []
        history_state = tuple()
        masks = tuple()

        for batch in reporter.measure_iter_time(iterator, "iter_time"):
            
        
            batch['mems'] = mems
            batch['complemented_version'] = complemented_version
            batch['masks'] = masks
            
            # obtain id and end point
            idlist = total_id_list[0]
            end_point = []
            for id in idlist:
                end_point.append(id2order[id])
            total_id_list = total_id_list[1:]
            batch['endpoint'] = end_point

            if concat_decoder:
                if len(history_state) != 0:
                    tmp_state_h = history_state[0].detach().clone()
                    tmp_state_c = history_state[1].detach().clone()
                    if len(history_state) != 0:
                        for i in range(len(end_point)):
                            if end_point[i] == 1:  
                                tmp_state_h[:, i, :] = 0.0
                                tmp_state_c[:, i, :] = 0.0
                    batch['history_state'] = (tmp_state_h, tmp_state_c)
                else:
                    batch['history_state'] = history_state
            else:
                batch['history_state'] = tuple()

            
            
            
            try:
                assert isinstance(batch, dict), type(batch)
                if no_forward_run:
                    all_steps_are_invalid = False
                    continue

                state = None
                if distributed:
                    state = "all_reduce"
                    with reporter.measure_time("all_reduce_time", verbose):
                        torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                        if iterator_stop > 0:
                            break

                with reporter.measure_time("forward_time", verbose), autocast(scaler is not None):
                    state = "forward"
                    batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                    # retval = model(**batch,)
                    retval = model(batch)

                    # Note(kamo):
                    # Supporting two patterns for the returned value from the model
                    #   a. dict type
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()
                        
                        #mems = retval["mems"]
                        #mem_lens = retval["mem_lens"]
                        # mems, mem_lens = cls.update_memory(mems, mem_lens)
                        

                    #   b. tuple or list type
                    else:
                        loss, stats, weight, feats, feats_lengths, mem, history_state, prev_masks = retval
                        if len(mem) != 0:
                            mems, current_length, index, masks = cls.update_memory(mems, mem.detach(), history_context_length, end_point, current_length, index, masks, prev_masks)
                        batch["feats"] = feats
                        batch["feats_lengths"] = feats_lengths
                        optim_idx = None

                    stats = {k: v for k, v in stats.items() if v is not None}
                    if ngpu > 1 or distributed:
                        # Apply weighted averaging for loss and stats
                        loss = (loss * weight.type(loss.dtype)).sum()

                        # if distributed, this method can also apply all_reduce()
                        stats, weight = recursive_average(stats, weight, distributed)

                        # Now weight is summation over all workers
                        loss /= weight
                    if distributed:
                        # NOTE(kamo): Multiply world_size because DistributedDataParallel
                        # automatically normalizes the gradient by world_size.
                        loss *= torch.distributed.get_world_size()

                    loss /= accum_grad

                    reporter.register(stats, weight)

                with reporter.measure_time("backward_time", verbose):
                    state = "backward"
                    if scaler is not None:
                        # Scales loss.  Calls backward() on scaled loss
                        # to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose
                        # for corresponding forward ops.
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    loss = None

                with reporter.measure_time("optim_time", verbose):
                    state = "optim"
                    if iiter % accum_grad == 0:
                        if scaler is not None:
                            # Unscales the gradients of optimizer's assigned params in-place
                            for iopt, optimizer in enumerate(optimizers):
                                if optim_idx is not None and iopt != optim_idx:
                                    continue
                                scaler.unscale_(optimizer)

                        # gradient noise injection
                        if grad_noise:
                            add_gradient_noise(
                                model,
                                reporter.get_total_count(),
                                duration=100,
                                eta=1.0,
                                scale_factor=0.55,
                            )

                        # compute the gradient norm to check if it is normal or not
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=grad_clip,
                            norm_type=grad_clip_type,
                        )
                        # if grad_norm > grad_clip:
                        #    logging.info(f"step: {cls.step}: grad norm({grad_norm}) exceed grad clip({grad_clip})")

                        # PyTorch<=1.4, clip_grad_norm_ returns float value
                        if not isinstance(grad_norm, torch.Tensor):
                            grad_norm = torch.tensor(grad_norm)

                        if not torch.isfinite(grad_norm):
                            logging.warning(f"The grad norm is {grad_norm}. Skipping updating the model.")

                            # Must invoke scaler.update() if unscale_() is used in the iteration
                            # to avoid the following error:
                            #   RuntimeError: unscale_() has already been called
                            #   on this optimizer since the last update().
                            # Note that if the gradient has inf/nan values,
                            # scaler.step skips optimizer.step().
                            if scaler is not None:
                                for iopt, optimizer in enumerate(optimizers):
                                    if optim_idx is not None and iopt != optim_idx:
                                        continue
                                    scaler.step(optimizer)
                                    scaler.update()

                        else:
                            all_steps_are_invalid = False
                            with reporter.measure_time("optim_step_time", verbose):
                                for iopt, (optimizer, scheduler) in enumerate(zip(optimizers, schedulers)):
                                    if optim_idx is not None and iopt != optim_idx:
                                        continue
                                    if scaler is not None:
                                        # scaler.step() first unscales the gradients of
                                        # the optimizer's assigned params.
                                        scaler.step(optimizer)
                                        # Updates the scale for next iteration.
                                        scaler.update()
                                    else:
                                        optimizer.step()
                                    if isinstance(scheduler, AbsBatchStepScheduler):
                                        scheduler.step()
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            optimizer.zero_grad()
                        # upate the step for update only
                        cls.step += 1

                    # Register lr and train/load time[sec/step],
                    # where step refers to accum_grad * mini-batch
                    reporter.register(
                        {
                            # f"optim{i}_lr{j}": pg["lr"]
                            "lr": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                    )

                duration = time.perf_counter() - start_time
                epoch_duration = time.perf_counter() - epoch_start_time
                if verbose:
                    reporter.register(
                        dict(train_time=duration),
                    )

                epoch_acc_num_feats += sum(batch["feats_lengths"]).item()
                reporter.register(  # fps is seq_len(fbank or raw wav)
                    # dict(fps=sum(batch["speech_lengths"]).item() / duration),
                    dict(fps=epoch_acc_num_feats / epoch_duration),
                )
                reporter.register(
                    dict(fpb=epoch_acc_num_feats * accum_grad / iiter),
                )
                start_time = time.perf_counter()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning(
                        f"OOM happended: Step({cls.step}) Batch({iiter}) state({state})"
                        f"Speech({batch['speech'].shape}) Text:({batch['text'].shape})\n<<<<{e}"
                    )
                    logging.info(f"CUDA: {torch.cuda.get_device_properties(0)}")
                    iterator.oom = True
                else:
                    logging.error(f"RuntimeError at Batch({iiter}) state({state}): {e}")
                    raise e

            

            # it is important to move oom here,
            # due to the memory deallocation scope for local variables
            # cf: https://pytorch.org/docs/stable/notes/faq.html
            if iterator.oom:
                ## oom on backward
                loss = None
                retval = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % accum_grad != 0:
                iiter += 1
                continue
            iiter += 1

            if cls.step <= 10 or cls.step % log_interval == 0:
                logging.info(reporter.log_message(cls.step, -log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)

            if cls.step % valid_log_interval == 0:
                cls.validate_one_epoch(
                    model=model,
                    iterator=cls.build_iterator(valid_iterator.build_iter(iepoch)),
                    reporter=reporter,
                    options=options,
                    distributed_option=distributed_option,
                    summary_writer=summary_writer,
                    id2order=valid_iterator.id2order
                )
        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        summary_writer,
        id2order,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()
        total_loss = 0.0
        total_utt = 0

        mems = tuple()
        
        history_context_length = options.history_context_length
        complemented_version = 0
        concat_decoder = options.concat_decoder
        masks = tuple()
        pos = tuple()
        total_id_list = iterator.iter._index_sampler
        current_length = 0
        index = []
        history_state = tuple()
        masks = tuple()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for batch in iterator:
            batch['mems'] = mems
            batch['complemented_version'] = complemented_version
            batch['masks'] = masks
            
            
            # obtain id and end point
            # batchsize = batch['speech'].size(0)
            idlist = total_id_list[0]
            end_point = []
            for id in idlist:
                end_point.append(id2order[id])
            total_id_list = total_id_list[1:]
            batch['endpoint'] = end_point


            if concat_decoder:
                if len(history_state) != 0:
                    tmp_state_h = history_state[0].detach().clone()
                    tmp_state_c = history_state[1].detach().clone()
                    if len(history_state) != 0:
                        for i in range(len(end_point)):
                            if end_point[i] == 1:  
                                tmp_state_h[:, i, :] = 0.0
                                tmp_state_c[:, i, :] = 0.0
                    batch['history_state'] = (tmp_state_h, tmp_state_c)
                else:
                    batch['history_state'] = history_state
            else:
                batch['history_state'] = tuple()
            
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # retval = model(**batch)
            retval = model(batch)
            if isinstance(retval, dict):
                stats = retval["stats"]
                weight = retval["weight"]
            else:
                #  _, stats, weight = retval
                _, stats, weight, feats, feats_lengths, mem, history_state, prev_masks = retval
                if len(mem) != 0:
                    mems, current_length, index, masks = cls.update_memory(mems, mem.detach(), history_context_length, end_point, current_length, index, masks, prev_masks)
                batch['feats'] = feats
                batch['feats_lengths'] = feats_lengths
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)
            loss = stats["loss"].item()
            total_loss += loss * weight.item()
            total_utt += weight.item()

            # reporter.register(stats, weight)
            # reporter.next()
            # break
        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        model.train()
        avg_loss = total_loss / total_utt
        logging.info(
            f"************** valid: {reporter.epoch}-th epoch: {cls.step}-th step: loss={avg_loss:.3f} **************"
        )
        # logging.info(reporter.log_message(0))
        if summary_writer is not None:
            reporter.tensorboard_add_scalar(summary_writer, -1)

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Optional[Path],
        summary_writer,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        import matplotlib

        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):

                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim > 3 or att_w.ndim == 1:
                        raise RuntimeError(f"Must be 2 or 3 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    if output_dir is not None:
                        p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(p)

                    if summary_writer is not None:
                        summary_writer.add_figure(f"{k}_{id_}", fig, reporter.get_epoch())

            reporter.next()
