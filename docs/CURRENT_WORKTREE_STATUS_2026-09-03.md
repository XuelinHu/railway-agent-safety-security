# Current Worktree Status - 2026-09-03

## Repository state

- Current branch: `master`.
- No branch was created during the interrupted snapshot attempt.
- No Git stash was created.
- The worktree retains all existing local modifications and generated results.
- At inventory time there were 22 modified tracked files and 70 untracked
  paths. These include the manuscript, figures, result artifacts, scripts, and
  API pilot outputs.

## Final manuscript and related references

- English manuscript PDF: `output/pdf/railway-safety-manuscript.pdf`
- Chinese manuscript PDF: `output/pdf/railway-safety-manuscript-zh.pdf`
- Title page PDF: `output/pdf/railway-safety-title-page.pdf`
- Liu and Luo (2025) PDF:
  `output/pdf/related_work/2025_Liu_Luo_LLM_KG_Accident_Reports.pdf`
- Josifoski et al. (2022) GenIE PDF:
  `output/pdf/related_work/2022_Josifoski_et_al_GenIE.pdf`

## Completed experiments

- D100 MBE external baseline completed all three seeds. Outputs are under
  `paper/results/external_baselines_d100/mbe/`.
- The validation-only API pilot completed for DeepSeek, Bailian Qwen-Plus, and
  sub2api GPT-5.4-mini. The strict pilot report is
  `paper/results/api_validation_pilot_20260903/PILOT_RESULTS.md`.
- The API pilot is exploratory only: it covers three validation documents and
  must not be placed in the main manuscript result table.

## Paused and not scheduled for recovery

- CPU CRF D100 external baseline: interrupted before a completed seed.
- Top-k D100 ablation: Top-2 job construction completed; its first QLoRA seed
  was interrupted. Top-8 and Top-16 did not start.
- D50 remains paused by instruction.
- No interrupted run directory is a valid completed experimental result.

## Process state

- No model training, inference, API pilot, Top-k, or Git snapshot process is
  running.
- The RTX 3090 is idle apart from driver memory.

## Next isolated workstream

Any future online-API work should begin in a new branch and use the existing
`scripts/run_api_validation_pilot.py` only as a baseline. It needs a second
stage with training-only KG candidate restriction, output budgets, source
evidence reconstruction, relation-signature validation, and final gates before
it can be compared with the main PGE protocol.
