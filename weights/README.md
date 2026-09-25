# Canonical R12 checkpoints

Upload the six **unchanged binary checkpoint files** from
`TWM_STATION_NEURAL_DEMO_R13_2026-09-25.zip` into this directory:

- `R12A_seed17_step5000.pt`
- `R12A_seed29_step5000.pt`
- `R12A_seed43_step5000.pt`
- `R12B_seed17.pt`
- `R12B_seed29.pt`
- `R12B_seed43.pt`

The runtime verifies every file against the SHA-256 values in
`SOURCE_PROVENANCE.json` **before** deserializing a checkpoint. A renamed,
modified, or wrong checkpoint fails closed and the app will not run.

These are the canonical R12 weights used by the frozen R13 rollout protocol.
No diagnostic R14–R20 checkpoint belongs here.
