#!/usr/bin/env python3
"""Run upstream SpERT with process-local modern Transformers compatibility."""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.optim import Optimizer


class HistoricalAdamW(Optimizer):
    """Transformers 4.1 AdamW, including the legacy ``correct_bias`` option."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("HistoricalAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    correction1 = 1.0 - beta1 ** state["step"]
                    correction2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(correction2) / correction1
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
                if group["weight_decay"] > 0.0:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        return loss


def install_compatibility() -> None:
    import transformers
    from transformers import PreTrainedModel

    transformers.AdamW = HistoricalAdamW
    if getattr(PreTrainedModel.save_pretrained, "_spert_compat", False):
        return
    original = PreTrainedModel.save_pretrained

    def save_pretrained_legacy_bin(self, save_directory, *args, **kwargs):
        kwargs.setdefault("safe_serialization", False)
        return original(self, save_directory, *args, **kwargs)

    save_pretrained_legacy_bin._spert_compat = True
    PreTrainedModel.save_pretrained = save_pretrained_legacy_bin


def install_config_dispatch(repo: Path) -> None:
    """Keep SpERT config dispatch inside the compatibility process.

    Upstream SpERT always launches each parsed configuration with the
    ``spawn`` multiprocessing context.  A spawned child imports ``spert.py``
    directly, so it cannot see the process-local Transformers compatibility
    symbols installed above.  The fresh baseline invokes one configuration
    per wrapper process; dispatching that configuration in the current
    process preserves its arguments while keeping the compatibility shim in
    scope.
    """

    config_reader = importlib.import_module("config_reader")
    expected = (repo / "config_reader.py").resolve()
    module_file = getattr(config_reader, "__file__", None)
    if module_file is None:
        raise RuntimeError("SpERT config_reader has no filesystem origin")
    actual = Path(module_file).resolve()
    if actual != expected:
        raise RuntimeError(f"unexpected SpERT config_reader: {actual} != {expected}")

    original_yield = config_reader._yield_configs

    def process_configs_in_current_process(target, arg_parser):
        args, _ = arg_parser.parse_known_args()
        for run_args, _run_config, _run_repeat in original_yield(arg_parser, args):
            # Upstream spawn pickles a separate Namespace for every repeat.
            # Preserve that argument isolation when dispatching inline.
            target(copy.deepcopy(run_args))

    process_configs_in_current_process._spert_compat = True
    config_reader.process_configs = process_configs_in_current_process


def parse_args() -> tuple[Path, list[str]]:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root / "tools/external-baselines/spert")
    parser.add_argument("spert_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.spert_args[1:] if args.spert_args[:1] == ["--"] else args.spert_args
    if not forwarded:
        parser.error("pass SpERT arguments after '--', for example: -- train --help")
    return args.repo.resolve(), forwarded


def main() -> int:
    repo, forwarded = parse_args()
    entrypoint = repo / "spert.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"SpERT entrypoint not found: {entrypoint}")
    install_compatibility()
    sys.path.insert(0, str(repo))
    install_config_dispatch(repo)
    os.chdir(repo)
    sys.argv = [str(entrypoint), *forwarded]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
