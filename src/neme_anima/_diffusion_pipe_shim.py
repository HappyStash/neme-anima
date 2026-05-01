"""Pre-train.py shim that patches diffusion-pipe's offloader to use blocking
GPU↔CPU transfers, then exec's train.py.

Why this exists
---------------
diffusion-pipe's block-swap offloader (utils/offloading.py) and its unsloth-
checkpoint helper (utils/unsloth_utils.py) move tensors between CPU and GPU
with ``tensor.to(<other_device>, non_blocking=True)``. Each non-blocking
GPU→CPU transfer requires a page-locked (pinned) host buffer for DMA. On
WSL2 the pinned-memory pool exposed to CUDA is much smaller than on native
Linux, and once it exhausts further allocations fail with a generic
``cudaErrorMemoryAllocation`` — surfaced by PyTorch as ``CUDA error: out of
memory`` even when the GPU has tens of gigabytes of free memory.

Empirically: on an RTX 4090 with 22 GB free, ``blocks_to_swap=16`` OOMs
mid-setup, with ``cuda_free=19.93 GB`` and ``torch_allocated=2.02 GB`` at
the moment of failure. Replacing every ``non_blocking=True`` GPU↔CPU
transfer in the offloader with the blocking equivalent (PyTorch falls back
to a synchronous pageable copy that doesn't need pinned memory) removes
the ceiling and training runs end-to-end.

The shim is invoked from ``training.build_launcher_argv`` whenever the
run TOML enables block_swap, i.e. it's gated on the user actually using
the offloader. Native-Linux users pay a tiny throughput cost (blocking
copies serialise with compute) but the offloader was already a 2–3×
slowdown vs. fitting on GPU, so the proportional cost is small.

Usage
-----
``deepspeed --num_gpus=1 _diffusion_pipe_shim.py --deepspeed --config <toml>``

The shim assumes its CWD is the diffusion-pipe checkout root (so
``utils.offloading``/``utils.unsloth_utils``/``train.py`` are importable).
The launcher always runs the trainer with ``cwd=diffusion_pipe_dir``.
"""
from __future__ import annotations

import os
import sys

# Make sure we can import diffusion-pipe's modules. The launcher cd's into
# diffusion_pipe_dir so '.' on sys.path is enough; insert it at position 0
# so we beat anything similarly named already imported.
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from utils import offloading  # noqa: E402
from utils import unsloth_utils  # noqa: E402


# ----- patch 1: setup-time bulk move (offloading.weights_to_device) ---------
#
# Used by ModelOffloader.prepare_block_devices_before_forward to move every
# block's weights to its target device exactly once at training start. The
# original loops with ``.to(device, non_blocking=True)``; on WSL2 the pinned
# pool exhausts somewhere around the 22nd–25th call.

def _weights_to_device_blocking(layer, device):
    for name, module in layer.named_modules():
        # Mirror the original LoRA exclusion: trainable adapter params must
        # not be moved to CPU because their gradients live on GPU and the
        # optimizer step would fail on a device mismatch.
        if device.type == 'cpu' and 'lora' in name:
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = module.weight.data.to(device)


offloading.weights_to_device = _weights_to_device_blocking


# ----- patch 2: per-step swap (offloading.swap_weight_devices_cuda) ---------
#
# Called once per block per training step to swap a CPU-resident block in
# and a GPU-resident one out. The original uses a dedicated stream + four
# non_blocking transfers per call; on WSL2 the pinned pool fills within a
# few steps. We replace with blocking transfers and skip the stream
# bookkeeping (it's only relevant when overlapping with compute, which
# blocking transfers preclude anyway).

def _swap_weight_devices_cuda_blocking(device, layer_to_cpu, layer_to_cuda):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if 'lora' in module_to_cuda_name:
            continue
        if hasattr(module_to_cuda, 'weight') and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if (
                module_to_cpu is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                weight_swap_jobs.append((
                    module_to_cpu, module_to_cuda,
                    module_to_cpu.weight.data, module_to_cuda.weight.data,
                ))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()

    # cuda → cpu (blocking)
    for m_cpu, m_cuda, cuda_view, _cpu_view in weight_swap_jobs:
        m_cpu.weight.data = cuda_view.data.to('cpu')
    # cpu → cuda (blocking)
    for m_cpu, m_cuda, cuda_view, _cpu_view in weight_swap_jobs:
        cuda_view.copy_(m_cuda.weight.data)
        m_cuda.weight.data = cuda_view

    torch.cuda.current_stream().synchronize()


offloading.swap_weight_devices_cuda = _swap_weight_devices_cuda_blocking


# ``Offloader.swap_weight_devices`` captured the original module-level
# function at class-definition time, so re-binding the module attr alone
# isn't enough — patch the method directly.
def _offloader_swap(self, block_to_cpu, block_to_cuda):
    if self.cuda_available:
        _swap_weight_devices_cuda_blocking(self.device, block_to_cpu, block_to_cuda)
    else:
        offloading.swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)


offloading.Offloader.swap_weight_devices = _offloader_swap


# ----- patch 3: unsloth checkpoint (utils.unsloth_utils) --------------------
#
# Activation checkpointing offloads the saved hidden state to CPU on the
# forward pass and brings it back on the backward pass. Same non_blocking
# pattern, same WSL2 failure mode. We rebuild the autograd Function with
# blocking transfers.

class _BlockingUnslothCheckpointer(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to('cpu')
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        # Imported lazily so the shim works even if deepspeed isn't on
        # sys.path at import time (it always is when launched via deepspeed,
        # but this keeps the failure mode obvious if someone runs the shim
        # standalone).
        from deepspeed.runtime.activation_checkpointing.checkpointing import (
            detach_variable,
        )

        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to('cuda').detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(inp.grad for inp in inputs)


@torch._disable_dynamo
def _unsloth_checkpoint_blocking(function, *args):
    return _BlockingUnslothCheckpointer.apply(function, *args)


unsloth_utils.unsloth_checkpoint = _unsloth_checkpoint_blocking


# ----- exec train.py --------------------------------------------------------
#
# We exec rather than import-and-call so train.py's ``if __name__ ==
# '__main__'`` guard fires and so its own argument-parsing sees argv as it
# would when invoked directly. ``sys.argv[0]`` is rewritten to ``train.py``
# so deepspeed's launcher (which inspects argv[0] for logging) reports
# the trainer name, not our shim.

train_py = os.path.join(os.getcwd(), 'train.py')
if not os.path.isfile(train_py):
    raise SystemExit(
        f"diffusion-pipe shim: train.py not found at {train_py!r}. The "
        f"launcher must run with cwd=<diffusion_pipe_dir>; check the "
        f"launcher invocation."
    )

sys.argv[0] = train_py
with open(train_py, 'r', encoding='utf-8') as fh:
    source = fh.read()
exec(compile(source, train_py, 'exec'), {'__name__': '__main__', '__file__': train_py})
