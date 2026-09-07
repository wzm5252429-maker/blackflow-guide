# Blackflow RL Training Report

Generated on 2026-09-07 (Asia/Shanghai).

## Result

The policy checkpoint was resumed from episode 1000 and trained for another 500
episodes, reaching episode 1500. The final checkpoint is stored at
`artifacts/blackflow_policy.pt` through Git LFS.

The simulation used the `synthetic` profile. Its rules have not yet been verified
against live game behavior, so this checkpoint should be treated as an experimental
training artifact rather than a production-ready policy.

## Evaluation

Both evaluations used 30 episodes, seed start 10000, 16 MCTS simulations, and CPU.

| Policy | Episode 1000 mean reward | Episode 1500 mean reward |
| --- | ---: | ---: |
| Random | 88.1147 | 88.1147 |
| Heuristic | 139.1117 | 139.1117 |
| MCTS + neural network | 129.9597 | 128.9370 |

At episode 1500, MCTS remained 46.5% above the random baseline and 7.3% below the
heuristic baseline. The additional 500 episodes did not improve the fixed-seed MCTS
score, so further training should first revisit the training schedule, replay data,
or simulator fidelity instead of blindly extending the same run.

## Reproduction

```powershell
py -m blackflow_rl train --profile synthetic --episodes 500 --resume artifacts/blackflow_policy.pt
py -m blackflow_rl evaluate --checkpoint artifacts/blackflow_policy.pt --episodes 30 --seed-start 10000 --simulations 16 --device cpu --profile synthetic
```

## Integrity

- Rules SHA-256: `e0b2e231b57984166923cb9f5e3ef967ea52b74ad6235f51918b9810812c0691`
- Environment SHA-256: `7327f79f6c7321ea2d6a6ebb0af304339e967314e4e03e17958773ad60bbbce6`
- Final episode: 1500
- Replay size: 20000
- Checkpoint size: approximately 379 MB
- Checkpoint SHA-256: `c39ba5a8ad118463a01620e640d21369959b794cf979fb7b66e2de60442e6d02`

The full continuation log is in `artifacts/training_1001_1500.log`, and the final
machine-readable evaluation is in `artifacts/evaluation_1500.json`.
