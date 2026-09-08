# Training-loss availability audit

The seed-2026 and seed-3407 QLoRA adapters for ADE and CoNLL04 contain
`training_metrics.json` with `mean_loss` and `final_loss`. They do not contain
the per-step loss sequence. Although `scripts/train_qlora.py` prints one loss
value every five optimizer steps, the formal matrix did not retain the
corresponding training stdout logs.

Consequently, a training-loss trajectory cannot be reconstructed faithfully
from the preserved artifacts. No interpolated or synthetic loss curve is used
in the manuscript. The endpoint summaries remain available for execution
audit, but mean loss and final loss are not plotted as if they were successive
training observations.
