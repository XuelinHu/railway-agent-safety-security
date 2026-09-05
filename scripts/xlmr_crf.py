#!/usr/bin/env python3
"""A small learnable linear-chain CRF for token-level BIO tagging.

The Hugging Face token-classification model remains the encoder/emission
checkpoint.  CRF parameters are saved in a separate file so the regular
``AutoModelForTokenClassification`` checkpoint remains loadable by existing
tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LinearChainCRF(nn.Module):
    """Linear-chain CRF with a boolean mask for special/padded tokens.

    ``mask`` may begin with false values (tokenizers commonly emit a special
    token before the first text token) and may contain trailing false values.
    Transitions are indexed as ``transitions[previous, current]``.
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        if num_labels < 2:
            raise ValueError("a CRF requires at least two labels")
        self.num_labels = int(num_labels)
        self.start_transitions = nn.Parameter(torch.zeros(num_labels))
        self.end_transitions = nn.Parameter(torch.zeros(num_labels))
        self.transitions = nn.Parameter(torch.zeros(num_labels, num_labels))

    def _validate_inputs(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [batch, sequence, labels]")
        if emissions.shape[-1] != self.num_labels:
            raise ValueError("emissions label dimension does not match the CRF")
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )
        else:
            mask = mask.to(device=emissions.device, dtype=torch.bool)
            if mask.shape != emissions.shape[:2]:
                raise ValueError("mask must have shape [batch, sequence]")
        if tags is not None:
            if tags.shape != emissions.shape[:2]:
                raise ValueError("tags must have shape [batch, sequence]")
            if tags.device != emissions.device:
                tags = tags.to(emissions.device)
            if torch.any(tags[mask] < 0) or torch.any(tags[mask] >= self.num_labels):
                raise ValueError("tags contain an out-of-range active label")
        return mask

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute log Z with masked positions skipped."""
        active_rows = mask.any(dim=1)
        if not active_rows.all():
            result = emissions.new_zeros(
                emissions.shape[0], dtype=self.start_transitions.dtype
            )
            if active_rows.any():
                result[active_rows] = self._log_partition(
                    emissions[active_rows], mask[active_rows]
                )
            return result
        batch_size, sequence_length, _ = emissions.shape
        neg_inf = torch.finfo(emissions.dtype).min
        alpha = torch.full(
            (batch_size, self.num_labels), neg_inf, dtype=emissions.dtype,
            device=emissions.device,
        )
        started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)
        for token_index in range(sequence_length):
            active = mask[:, token_index]
            emission = emissions[:, token_index]
            from_start = self.start_transitions + emission
            from_previous = torch.logsumexp(
                alpha.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1
            ) + emission
            alpha = torch.where(
                (active & started).unsqueeze(1),
                from_previous,
                torch.where(active.unsqueeze(1), from_start, alpha),
            )
            started = started | active
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def _gold_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Score gold paths while skipping inactive special/padded tokens."""
        batch_size, sequence_length, _ = emissions.shape
        score = torch.zeros(batch_size, dtype=emissions.dtype, device=emissions.device)
        previous = torch.zeros(batch_size, dtype=torch.long, device=emissions.device)
        started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)
        for token_index in range(sequence_length):
            active = mask[:, token_index]
            current = tags[:, token_index]
            emission = emissions[:, token_index].gather(1, current.unsqueeze(1)).squeeze(1)
            start_score = self.start_transitions[current]
            transition_score = self.transitions[previous, current]
            score = score + torch.where(
                active,
                emission + torch.where(started, transition_score, start_score),
                torch.zeros_like(score),
            )
            previous = torch.where(active, current, previous)
            started = started | active
        score = score + self.end_transitions[previous]
        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return ``-log p(tags | emissions)`` for a batch."""
        tags = tags.to(device=emissions.device, dtype=torch.long)
        mask = self._validate_inputs(emissions, tags, mask)
        active_rows = mask.any(dim=1)
        if active_rows.any():
            active_emissions = emissions[active_rows]
            active_tags = tags[active_rows]
            active_mask = mask[active_rows]
            active_loss = self._log_partition(
                active_emissions, active_mask
            ) - self._gold_score(active_emissions, active_tags, active_mask)
            loss = emissions.new_zeros(
                emissions.shape[0], dtype=active_loss.dtype
            )
            loss[active_rows] = active_loss
        else:
            # Keep the zero loss connected to the encoder graph so a batch of
            # empty chunks can still participate in gradient accumulation.
            loss = emissions.sum(dim=(1, 2)) * 0.0
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction != "mean":
            raise ValueError(f"unknown CRF reduction: {reduction}")
        return loss[active_rows].mean() if active_rows.any() else loss.sum()

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the log partition, or NLL when gold tags are supplied."""
        if tags is None:
            mask = self._validate_inputs(emissions, None, mask)
            return self._log_partition(emissions, mask)
        tags = tags.to(device=emissions.device, dtype=torch.long)
        mask = self._validate_inputs(emissions, tags, mask)
        return self.neg_log_likelihood(emissions, tags, mask)

    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return best-path label ids, with zero at inactive positions."""
        mask = self._validate_inputs(emissions, None, mask)
        paths = torch.zeros(
            emissions.shape[:2], dtype=torch.long, device=emissions.device
        )
        for batch_index in range(emissions.shape[0]):
            positions = mask[batch_index].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            compact = emissions[batch_index, positions]
            score = self.start_transitions + compact[0]
            backpointers: list[torch.Tensor] = []
            for token_index in range(1, compact.shape[0]):
                candidates = score.unsqueeze(1) + self.transitions
                best_score, previous = candidates.max(dim=0)
                score = best_score + compact[token_index]
                backpointers.append(previous)
            current = int((score + self.end_transitions).argmax())
            decoded = [current]
            for previous in reversed(backpointers):
                current = int(previous[current])
                decoded.append(current)
            decoded.reverse()
            paths[batch_index, positions] = torch.tensor(
                decoded, dtype=torch.long, device=emissions.device
            )
        return paths

    def save(self, path: str | Path) -> None:
        """Save CRF parameters independently of the HF checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"num_labels": self.num_labels, "state_dict": self.state_dict()}, target
        )

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "LinearChainCRF":
        payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
        crf = cls(int(payload["num_labels"]))
        crf.load_state_dict(payload["state_dict"])
        return crf
