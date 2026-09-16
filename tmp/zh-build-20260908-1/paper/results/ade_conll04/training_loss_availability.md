# Training-loss availability audit

The primary seed-42 EAE and HRGE training runner retained one stdout loss
observation every five optimizer steps for ADE and CoNLL04. The extraction
script matches each logged block to its adapter `training_metrics.json` by the
exact `started_at_utc` value and verifies its total-step and final-loss
metadata. The source-log SHA-256 and per-adapter observation counts are stored
in `training_loss_seed42_provenance.json`; the extracted observations are in
`training_loss_seed42.csv`.

The logging interval means that four optimizer steps between adjacent points
were never recorded. They are not reconstructed or interpolated. The plotted
trailing mean is computed only from observed values, while raw observations
remain visible. SOE and the two additional training repetitions do not have
equivalently attributable step histories and are therefore excluded from the
loss figure; their mean/final loss values are not presented as trajectories.

For future runs, `scripts/train_qlora.py` saves every optimizer-step loss to
`training_loss.csv` directly. This avoids dependence on captured stdout and
supports complete, traceable curves.
