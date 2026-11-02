# PowerFORM

Code, run records and checkpoints for

> **PowerFORM: A General-Purpose Foundation Model for Power Systems.**
> Seungchan Jo, Gyu-Sub Lee. NeurIPS 2026 Workshop on AI Foundations for Power Grids.

PowerFORM's hidden state is indexed by physical quantity. Bus voltage, bus
injection and line flow each have a stream; FlowNet and BusNet alternate between
them, and an AC power flow closes every forward pass. A new decision attaches to the
stream of its physical quantity: reconfiguration adds a line-status track to the line
stream without changing the backbone.
[`powerform/ARCHITECTURE.md`](powerform/ARCHITECTURE.md) gives the network in full.

## Setup

Python 3.12 (tested with 3.12.7).

```
python3.12 -m venv .venv         # Windows: py -3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1, or in Git Bash source .venv/Scripts/activate
pip install -r requirements.txt
```

Every command below assumes the environment is active and runs from the repository
root. Commands read `results/` and `checkpoints/` and write to `outputs/`.

**GPU.** On Windows, `requirements.txt` installs a CPU-only torch. For a CUDA build:
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`.
Trainers default to `--device auto`, which takes CUDA when present. Scoring needs
no GPU. Set `FORCE_CPU=1` to make `auto` choose the CPU (in PowerShell, `$env:FORCE_CPU = 1`).

**CPU threads.** The CPU times below are for one thread:
`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (in PowerShell, `$env:OMP_NUM_THREADS = 1;
$env:MKL_NUM_THREADS = 1`).

## Data

Corpora live under one data home: `$POWERFORM_DATA`, or `datasets/` by default.
[`powerform/data/corpus/README.md`](powerform/data/corpus/README.md) has the full
instructions and checksums.

```
python -m powerform.data.corpus.fetch_opfdata --all --variant both
python -m powerform.data.corpus.fetch_test_split --case case2000 --variant fulltop
python -m powerform.data.corpus.fetch_test_split --case case6470 --variant fulltop
```

The Table 1 corpus (`case500_goc`, 30k scenarios) is generated with `gridfm-datakit`,
which takes about 12 hours and needs Julia, PowerModels and IPOPT (section 4 of that
README). IPOPT solves are not guaranteed bit-reproducible across machines, so a
regenerated corpus may move Table 1's numbers slightly.

## Checkpoints

Checkpoints are assets of the
[`v1.0` release](https://github.com/AnnualResearcher/PowerFORM/releases/tag/v1.0).
This downloads them to `checkpoints/` and verifies each SHA-256 against
`checkpoints.json`:

```
python scripts/download_checkpoints.py --list
python scripts/download_checkpoints.py                  # all, about 5 GB
python scripts/download_checkpoints.py --group table2   # one group
```

| Group | Paper item |
|---|---|
| `table1` | Table 1 and the App. D in-distribution table |
| `table2` | Table 2 and the App. D out-of-distribution table |
| `fig1` | the generalists Figure 1 starts from |
| `reconfig` | final Figure 1 models with physics maps, read by the App. C infeasible split |

## Reproducing the paper

`results/` holds the record behind every number, so the figure and the windows need
no checkpoints or data. [`results/README.md`](results/README.md) indexes it.

### Figure 1 and the reconfiguration numbers

```
python -m powerform.experiments.figure       # outputs/figures/figure1.{pdf,png}
python -m powerform.experiments.windows      # step windows and first crossings
python -m powerform.experiments.seed_table   # per-seed table
python -m powerform.eval.infeasible_split --arm pretrained --seed 0   # App. C split; needs `reconfig`
```

To retrain, fetch the `fig1` group first. An arm took about 13 h at 20,000 steps on
one GPU with physics maps and about 7 h with generic attention. Physics maps use seeds
0-5 and generic attention seeds 0-2.

```
python -m powerform.training.reconfiguration --maps physics --arm pretrained --seed 0
python -m powerform.training.reconfiguration --maps physics --arm scratch --seed 0
python -m powerform.training.reconfiguration --maps generic --arm pretrained --seed 0
```

The defaults are the paper's settings (App. C). `python -m powerform.experiments.reference
--split val` recomputes the greedy reference in about 3 minutes.

### Table 1 and the App. D in-distribution table

Needs the Table 1 corpus and the `table1` group. Re-scoring reads only the 500 hold-out
scenarios (one thread: about 16 min for the physics row, 7 min for MLP):

```
python -m powerform.training.case500 --eval-only checkpoints/case500_physics_seed0.pt
python -m powerform.training.case500 --eval-only checkpoints/case500_generic_no_type_masking_seed0.pt
python -m powerform.training.case500 --eval-only checkpoints/case500_mlp_seed0.pt
python powerform/tests/test_table1.py --summarize outputs/case500/rescore
```

Each row is a 3-seed mean. `--summarize` compares your records with `results/`: exit 0
means every row was run and agrees, 1 a disagreement, 2 no disagreement but some not run.

Violations per scenario of the PowerFORM rows, about 15 min per checkpoint (the baseline
rows of this column ship as records):

```
python -m powerform.eval.violations --ckpt checkpoints/case500_physics_seed0.pt --device cpu
python powerform/tests/test_violations.py --summarize outputs/table1
```

To retrain (full corpus, about 13 GB of host memory; 5-9 h per seed on one GPU for
the two PowerFORM rows):

| Row | Flags of `python -m powerform.training.case500` |
|---|---|
| PowerFORM, physics maps | `--arch physics --seed S` |
| PowerFORM, generic attention | `--arch generic --seed S` |
| F1: no type masking | `--arch generic --no-type-masking --seed S` |
| F2: merged readout | `--arch generic --merged-readout --seed S` |
| F3: final-layer supervision | `--arch generic --loss final-only --seed S` |
| MLP, GNN | `--arch mlp --seed S`, `--arch gnn --seed S` |
| GridSFM, decision and state regression | `--arch gridsfm --seed S`, `--arch gridsfm --recipe state --seed S` |

The graphkit rows run in their own environment and need a CUDA GPU to train; see
[`powerform/model/baselines/README.md`](powerform/model/baselines/README.md).

### Table 2 and the App. D out-of-distribution table

Needs OPFData and the `table2` group. Each PowerFORM row is a mean over seeds 0-2. Repeat
the checkpoint commands for each seed, and with `opf_generalist_generic_seed*.pt` for App. D:

```
python -m powerform.eval.opf_eval --ckpt checkpoints/opf_generalist_physics_seed0.pt --eval case57,case118,case500,case30,case2000,case6470 --device cpu
python -m powerform.eval.opf_eval --ckpt checkpoints/opf_generalist_physics_seed0.pt --eval case57,case118,case500 --variant n1 --device cpu
python -m powerform.eval.opf_eval --baseline ed --eval case57,case118,case500,case30,case2000,case6470 --device cpu
python -m powerform.eval.opf_eval --baseline ed --eval case57,case118,case500 --variant n1 --device cpu
python powerform/tests/test_opf_eval.py --summarize outputs/table2
```

`--summarize` reads and exits as for Table 1, for the cells of `results/table2/`.

One-thread CPU time per cell, 750 scenarios:

| Grid | PowerFORM | ED |
|---|---|---|
| `case30`, `case57` | 1-2 min | seconds |
| `case118` | 2-4 min | seconds |
| `case500` | 26-38 min | 11 min |
| `case2000` | 1-2 h | 1 min |
| `case6470` | about 4 h | 3 min |

The GridSFM backbone row of Table 2 ships as records only, one seed
(`results/table2/gridsfm_seed0_*`).

To retrain a generalist (2.4-5.9 h per seed on one GPU), for `--maps physics` or `generic`:

```
python -m powerform.training.generalist --maps physics --seed 0              # Table 2, App. D
python -m powerform.training.generalist --maps physics --seed 0 --ed-prior   # Figure 1 start
```

### Agreement with the paper

Re-scored numbers agree with `results/` to the printed digits and every count is identical
when scored on the corpora the records came from. The last floating-point bits depend
on the torch build, so the `--summarize` tools allow for them.

For a check in minutes, `--limit 5` on `case500 --eval-only` and `--n 4` on `opf_eval` and
`violations` score a subset. These write records of their own (`*_first5.json`,
`*_n4.json`, `violations_first4.json`), which `--summarize` does not read.

Training on a GPU does not repeat bit for bit, even with the same code and seed, so a
retrained checkpoint will differ from the shipped one and its scores will move.
Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` and calling
`torch.use_deterministic_algorithms(True)` makes a run repeat on one machine. The trainers
do not set it, and it does not recover the shipped weights, which were trained without it.

## Names in the code

| Paper | Code |
|---|---|
| PowerFORM | `PowerForm`, `powerform/model/powerform.py` |
| physics maps (FlowNet, BusNet) | `maps="physics"`; `powerform/model/flownet.py`, `powerform/model/busnet.py` |
| generic attention | `maps="generic"`; `powerform/model/generic.py` |
| line-status track | `LineStatusTrack`, `ReconfigurationNet`; `powerform/model/line_track.py` |
| pretrained, from scratch | `--arm pretrained`, `--arm scratch` |
| infeasible fraction | `infeasible_fraction` |
| ED-prior input columns | `--ed-prior` (`n_cost_feats` 12; otherwise 10 for generalists, 0 for Table 1) |

## Tests

Each test is a plain script, `python powerform/tests/<name>.py`, that exits non-zero on a
failure and ends by saying what ran and what was skipped. A check whose data or
checkpoints are missing is skipped with the command that fetches them, so run the tests
again after downloading for full coverage. `test_data.py` and `test_violations.py` take
`--deep` for their slow checks: the App. B numbers (reads two columns of the full bus
table) and a full violations row (about 15 min).

| Test | Checks |
|---|---|
| `test_standalone.py` | every module imports with only this repository on the path |
| `test_model.py` | parameter counts, batching, closed line equals deleted line, identity of the line-status track at start, ablation flags change the function |
| `test_data.py` | corpus verification, input layouts |
| `test_figures.py` | Figure 1 and the reconfiguration numbers regenerate from `results/` |
| `test_opf_eval.py`, `test_violations.py` | scoring detects perturbed weights and inputs; printed cells match `results/` |
| `test_table1.py` | every printed Table 1 and App. D row matches `results/table1/`; spot re-scores; the vendored GridSFM files match upstream |

## Citation

```bibtex
@inproceedings{jo2026powerform,
  title     = {{PowerFORM}: A General-Purpose Foundation Model for Power Systems},
  author    = {Jo, Seungchan and Lee, Gyu-Sub},
  booktitle = {NeurIPS 2026 Workshop on AI Foundations for Power Grids},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). Third-party code and data keep their own licenses:

- GridSFM (Microsoft, MIT): 15 modules vendored unchanged under
  `powerform/third_party/gridsfm/` from
  [microsoft/GridSFM](https://github.com/microsoft/GridSFM) at commit
  `1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`, with its `LICENSE` and a `NOTICE`.
- gridfm-graphkit 0.8.1 (Apache-2.0,
  [gridfm/gridfm-graphkit](https://github.com/gridfm/gridfm-graphkit) at
  `b3d663b62179222c1ebec00ee29f67ea50e68c0b`): installed, not vendored.
- gridfm-datakit 1.0.5 (Apache-2.0): used to generate the Table 1 corpus, not vendored.
- OPFData (Lovett et al., 2024) and PGLib-OPF case files: CC BY 4.0.
