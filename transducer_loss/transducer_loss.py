import torch
# import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.nn import Module
from torch.utils.cpp_extension import load

import socket
import tempfile
import os
import logging
import pathlib
build_dir = os.path.dirname(os.path.abspath(__file__)) + "/rnntLoss_cpp"
os.makedirs(build_dir, exist_ok=True)
logging.info("Compiling C++ code to: {}".format(build_dir))
# logging.info(os.environ["INCLUDEPATH"])
if "INCLUDEPATH" not in os.environ:
    os.environ["INCLUDEPATH"] = ''
transducer_loss = load(
    name='transducer_loss',
    extra_include_paths=[os.environ["INCLUDEPATH"] + ":/usr/local/cuda/include"],
    sources=[os.path.join(os.path.dirname(__file__), source_file) for source_file in ["transducer_loss.cpp", "detail/gpu_rnnt.cu"]],
    build_directory=build_dir,
    verbose=True
)
logging.info("successfully build rnnt_loss")
with open(build_dir + '/complete', 'w') as f:
    f.write("DONE")


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, reduction, fastemit_lambda):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        fastemit_lambda: Regularization parameter for FastEmit (https://arxiv.org/pdf/2010.11148.pdf)
        """
        # only run in cuda
        assert acts.is_cuda is True
        minibatch_size = acts.size(0)

        certify_inputs(acts, labels, act_lens, label_lens)

        """
        inplace_logsoftmax = False
        # NOTE manually done log_softmax for CPU version,
        # log_softmax is computed within GPU version.
        if not acts.is_cuda:
            # in-place logsoftmax
            if inplace_logsoftmax is True:
                max_acts, _ = acts.max(dim=3, keepdim=True)
                acts.sub_(max_acts)
                acts.exp_()
                logsumexp = acts.sum(dim=3, keepdim=True)
                logsumexp.log_()
                acts.log_()
                acts.sub_(logsumexp)
            else:
                acts = torch.nn.functional.log_softmax(acts, -1)
        """

        # loss_func = warp_rnnt.gpu_rnnt if is_cuda else warp_rnnt.cpu_rnnt
        loss_func = transducer_loss.gpu_rnnt
        grads = torch.zeros(minibatch_size, dtype=acts.dtype)
        # grads = torch.zeros_like(acts)
        costs = torch.zeros(minibatch_size, dtype=acts.dtype)
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  grads,
                  blank,
                  fastemit_lambda,
                  0)

        if reduction in ['sum', 'mean']:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == 'mean':
                costs /= minibatch_size
                # grads /= minibatch_size
                acts /=  minibatch_size

        costs = costs.to(acts.device)

        ctx.grads = acts

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        # return ctx.grads.mul_(grad_output), None, None, None, None, None, None
        return ctx.grads, None, None, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'
    """
    def __init__(self, blank=0, fastemit_lambda=0.000, reduction='mean'):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.loss = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        labels = labels.int()
        act_lens = act_lens.int()
        label_lens = label_lens.int()

        return self.loss(acts, labels, act_lens, label_lens, self.blank, self.reduction, self.fastemit_lambda)


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))

def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))

def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))

def certify_inputs(log_probs, labels, lengths, label_lengths):
    # check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")

