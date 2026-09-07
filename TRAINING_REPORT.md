# Blackflow RL Training Report

Generated on 2026-09-07 (Asia/Shanghai).

## Result

The policy checkpoint was resumed across three continuation runs and trained to
episode 3500. The final checkpoint is stored at
`artifacts/blackflow_policy.pt` through Git LFS.

For the 2026-09-07 episode-3500 delivery, the full checkpoint is also published as
the `blackflow_policy_episode_3500.pt` asset on the `policy-episode-3500` GitHub
release. This release copy provides a fallback when GitHub smart-HTTP/LFS transport
is unavailable.

The simulation used the `synthetic` profile. Its rules have not yet been verified
against live game behavior, so this checkpoint should be treated as an experimental
training artifact rather than a production-ready policy.

## Evaluation

Both evaluations used 30 episodes, seed start 10000, 16 MCTS simulations, and CPU.

| Policy | Episode 1000 | Episode 1500 | Episode 2500 | Episode 3500 |
| --- | ---: | ---: | ---: | ---: |
| Random | 88.1147 | 88.1147 | 88.1147 | 88.1147 |
| Heuristic | 139.1117 | 139.1117 | 139.1117 | 139.1117 |
| MCTS + neural network | 129.9597 | 128.9370 | 134.6653 | 134.1067 |

Episode 2500 is the best fixed-seed checkpoint observed so far. Episode 3500 remains
52.2% above the random baseline and 3.6% below the heuristic baseline, but it declined
0.4% from episode 2500. This suggests the current training configuration is near a
plateau; further improvement should revisit the learning schedule, replay sampling,
or simulator fidelity instead of relying only on more episodes.

## Reproduction

```powershell
py -m blackflow_rl train --profile synthetic --episodes 1000 --resume artifacts/blackflow_policy.pt
py -m blackflow_rl evaluate --checkpoint artifacts/blackflow_policy.pt --episodes 30 --seed-start 10000 --simulations 16 --device cpu --profile synthetic
```

## Integrity

- Rules SHA-256: `e0b2e231b57984166923cb9f5e3ef967ea52b74ad6235f51918b9810812c0691`
- Environment SHA-256: `7327f79f6c7321ea2d6a6ebb0af304339e967314e4e03e17958773ad60bbbce6`
- Final episode: 3500
- Replay size: 20000
- Checkpoint size: approximately 379 MB
- Checkpoint SHA-256: `49098dfa0da6ab7b9d312ed03cc30dfddeba36926bb63cc4b92b4e045837d320`

Continuation logs and machine-readable evaluations are retained in `artifacts/` for
episodes 1500, 2500, and 3500.
