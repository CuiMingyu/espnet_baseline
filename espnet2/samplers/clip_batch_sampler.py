from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union
from itertools import cycle
import random

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class ClipBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        # self.batch_bins = batch_bins
        self.batch_size = batch_size
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        #keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        # keys = sorted(first_utt2shape, key=lambda k: (first_utt2shape[k][0], utt2shapes[1][k][0]))
        keys = list(first_utt2shape.keys())

        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")
        if padding:
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        # Decide session to keys
        # session means: audio01
        # audio   means: audio01_01, ...
        # Calculate total bins of each audio
        # session  = {
        # session_id : {'audio' : [audio01_01, ...], 'bins': totalbatchbins} 
        
        sessions = {}
        scale = 100.0 / 16000
        for key in keys:
            session_id = key.split('_')[0]
            if session_id not in sessions.keys():
                sessions[session_id] = {'audio': [key], 
                                        'bins': utt2shapes[0][key][0] * 
                                                utt2shapes[1][key][0] * scale}
            else:
                sessions[session_id]['audio'].append(key)
                sessions[session_id]['bins'] += utt2shapes[0][key][0] * utt2shapes[1][key][0] * scale
        

        batchlist = [[] for i in range(batch_size)]
        count = 0
        length = [0] * batch_size

        # if shuffle
        keys = list(sessions.keys())
        random.shuffle(keys)


        for key in keys:
            if count < batch_size:
                batchlist[count] += sessions[key]['audio']
                length[count] += sessions[key]['bins']
                count += 1
            else:
                min_value = min(length)
                idx = length.index(min_value)
                batchlist[idx] += sessions[key]['audio']
                length[idx] += sessions[key]['bins']
        

        # standard each batch
        max_batch_length = 0 
        for batch in batchlist:
            if len(batch) > max_batch_length:
                max_batch_length = len(batch)
        
        for batch in batchlist:
            bs = cycle(batch)
            while len(batch) < max_batch_length:
                batch.append(next(bs))
        
        batches = {}
        for current_batch in batchlist:
            for key in range(len(current_batch)):
                if key not in batches.keys():
                    batches[key] = [current_batch[key]]
                else:
                    batches[key].append(current_batch[key])


        if len(batches) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        batch_sizes = []
        keys = []
        for batch in batches.values():
            batch_sizes.append(len(batch))
            keys += batch
        
        # if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
        #     for i in range(batch_sizes.pop(-1)):
        #         batch_sizes[-(i % len(batch_sizes)) - 1] += 1


        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )

                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)



    #     # Decide batch-sizes
    #     batch_sizes = []
    #     current_batch_keys = []
    #     current_bins = []
    #     count_bins = []
    #     for key in keys:
    #         current_batch_keys.append(key)
    #         # shape: (Length, dim1, dim2, ...)
    #         if padding:
    #             for d, s in zip(utt2shapes, shape_files):
    #                 if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
    #                     raise RuntimeError(
    #                         "If padding=True, the "
    #                         f"feature dimension must be unified: {s}",
    #                     )
    #             # note that for Gigaspeech, sampling rate = 16k
    #             if False:
    #                 bins = sum(
    #                     len(current_batch_keys) * sh[key][0] * d
    #                     for sh, d in zip(utt2shapes, feat_dims)
    #                 )
    #             else:
    #                 # bins = len(current_batch_keys) * utt2shapes[0][key][0] * utt2shapes[1][key][0]
    #                 current_bins.append(utt2shapes[0][key][0] * utt2shapes[1][key][0])
    #                 max_bins = len(current_batch_keys) * max(current_bins) 
    #                 # convert bins to mbsize * T * U
    #                 # where T: number of frame, U: number of BPE
    #                 # by multiply the scale: 100.0(frame shift)/16000(sample rate)
    #                 # bins is number of frames not number of samples
    #                 scale = 100.0 / 16000
    #                 bins = max_bins * scale

    #         else:
    #             bins = sum(
    #                 np.prod(d[k]) for k in current_batch_keys for d in utt2shapes
    #             )

    #         if (bins > batch_bins or len(current_batch_keys) > max_batch_size) and len(current_batch_keys) >= min_batch_size:
    #             batch_sizes.append(len(current_batch_keys))
    #             current_batch_keys = []
    #             current_bins = []
    #             count_bins.append(bins)
                
    #         # audio01_01 audio01_02 audio01_03
    #         # audio01_04 audio01_05 
    #         # minibatch=3
    #         # audio01_01 audio02_01 audio03_01
    #         # audio01_02 audio02_02 audio03_02
    #         # audio04_01 audio02_03 audio03_03 audio05_01
            
    #     else:
    #         if len(current_batch_keys) != 0 and (
    #             not self.drop_last or len(batch_sizes) == 0
    #         ):
    #             batch_sizes.append(len(current_batch_keys))

    #     if len(batch_sizes) == 0:
    #         # Maybe we can't reach here
    #         raise RuntimeError("0 batches")

    #     # If the last batch-size is smaller than minimum batch_size,
    #     # the samples are redistributed to the other mini-batches
    #     if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
    #         for i in range(batch_sizes.pop(-1)):
    #             batch_sizes[-(i % len(batch_sizes)) - 1] += 1

    #     if not self.drop_last:
    #         # Bug check
    #         assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

    #     # Set mini-batch
    #     self.batch_list = []
    #     iter_bs = iter(batch_sizes)
    #     bs = next(iter_bs)
    #     minibatch_keys = []
    #     for key in keys:
    #         minibatch_keys.append(key)
    #         if len(minibatch_keys) == bs:
    #             if sort_batch == "descending":
    #                 minibatch_keys.reverse()
    #             elif sort_batch == "ascending":
    #                 # Key are already sorted in ascending
    #                 pass
    #             else:
    #                 raise ValueError(
    #                     "sort_batch must be ascending"
    #                     f" or descending: {sort_batch}"
    #                 )

    #             self.batch_list.append(tuple(minibatch_keys))
    #             minibatch_keys = []
    #             try:
    #                 bs = next(iter_bs)
    #             except StopIteration:
    #                 break

    #     if sort_batch == "ascending":
    #         pass
    #     elif sort_batch == "descending":
    #         self.batch_list.reverse()
    #     else:
    #         raise ValueError(
    #             f"sort_batch must be ascending or descending: {sort_batch}"
    #         )

    # def __repr__(self):
    #     return (
    #         f"{self.__class__.__name__}("
    #         f"N-batch={len(self)}, "
    #         f"batch_bins={self.batch_bins}, "
    #         f"sort_in_batch={self.sort_in_batch}, "
    #         f"sort_batch={self.sort_batch})"
    #     )

    # def __len__(self):
    #     return len(self.batch_list)

    # def __iter__(self) -> Iterator[Tuple[str, ...]]:
    #     return iter(self.batch_list)
