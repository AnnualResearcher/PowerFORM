# results

The records behind the paper's numbers: settings and values only. `SHA256SUMS` hashes
every file here. Checkpoints are in the GitHub release, not in this directory.

| Path | Paper item |
|---|---|
| `table1/case500_<row>_seed{0,1,2}.json` | Table 1 and the App. D in-distribution table: cost MAPE, channel errors, parameters, samples, training time, per-scenario cost |
| `table1/violations.json` | Table 1, violations per scenario, every row |
| `table2/{physics,generic}_seed{0,1,2}_<grid>_<variant>.json` | Table 2, App. D out-of-distribution table |
| `table2/ed_<grid>_<variant>.json` | Table 2, ED reference row |
| `table2/gridsfm_seed0_<grid>_<variant>.json` | Table 2, GridSFM backbone row |
| `generalists/opf_generalist_{physics,generic}{,_ed}_seed{0,1,2}.json` | training records of the generalists; `_ed` adds the ED-prior input columns and starts Figure 1 |
| `reconfiguration/{physics,generic}_{pretrained,scratch}_seed{s}.json` | Figure 1 curves (physics maps seeds 0-5, generic attention seeds 0-2) |
| `reconfiguration/{windows,crossings,seed_table}_{physics,generic}.json` | Sec. 4.2 and App. C step windows, first crossings, per-seed table |
| `reconfiguration/infeasible_split_physics_{pretrained,scratch}_seed{s}.json` | App. C: infeasible decisions split into divergence and voltage-limit violations |
| `reconfiguration/reference_val.json` | App. C: greedy reference and random three-line control per validation scenario |
| `reconfiguration/base_feasibility.json` | App. C: post-contingency do-nothing states |
| `reconfiguration/task_entropy.json` | App. C: how much the best opening depends on the scenario |
| `reconfiguration/islanding_prior.json` | App. C: how often a random three-line opening islands `case57` |
| `figures/figure1.{pdf,png}` | Figure 1 |
| `long_run/case500_{physics,gridsfm_decision}_steps{60000,120000}_seed0.json` | App. D, Figure 2: ends of the 60k- and 120k-step runs (cost MAPE, channel errors, per-scenario cost, training time) |
| `long_run/milestones_{physics,gridsfm}_steps120000_seed0.json` | Figure 2, lines: every 5k-step milestone of the 120k-step runs, scored on the Table 1 hold-out |
| `long_run/curve_{physics,gridsfm}_steps120000_seed0.json` | Figure 2, lower panel: mean training loss and learning rate every 200 steps |
| `reconfiguration/gridsfm_{pretrained,scratch}_seed{0,1,2}.json` | App. D, Figure 3: reconfiguration with the GridSFM backbone and the attached gate |
| `reconfiguration/settled_steps.json` | App. D: settled-step table of the three backbones |
| `figures/figure{2,3}.{pdf,png}` | Figures 2 and 3 |

`<variant>` is `fulltop` (base topology) or `n1` (unseen N-1 topology), OPFData's names.

The run records under `long_run/` are the files the runs wrote. The milestone and log files
merge the runs' JSON lines, values untouched, as `powerform.experiments.long_run_records` does.

`base_feasibility.json`, `task_entropy.json` and `islanding_prior.json` ship as records
only: no script in this repository regenerates them.
