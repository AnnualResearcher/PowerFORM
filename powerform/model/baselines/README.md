# Table 1 baselines

The comparison rows of Table 1 (in-distribution OPF on `case500_goc`, 3 seeds each).
Every row is trained and re-scored through one entry point,
`python -m powerform.training.case500`, on the same corpus and the same 500 hold-out
scenarios (ids 29,500-29,999). The decision rows share the PowerFORM rows' budget of
20,000 steps at batch 4 and their terminal power-flow evaluation.

| row | module | recipe | params | samples |
|---|---|---|---|---|
| MLP, decision | `mlp.py` | decision MSE + balance, final output only | 15,066,558 | 80,000 |
| GNN, decision | `gnn.py` | same | 15,041,502 | 80,000 |
| GridSFM, decision | `gridsfm.py` (`DecisionHeadGridSFM`) | same | 15,103,808 | 80,000 |
| GridSFM, state regression | `gridsfm.py` + vendored GridSFM | GridSFM's fine-tuning loop, 3 epochs from random init | 15,148,227 | 87,711 |
| graphkit, decision | `graphkit.py` | decision MSE + balance, final output only | 20,069,187 | 80,000 |
| graphkit, state regression | `graphkit.py` + graphkit's CLI | graphkit's Lightning recipe, 3 epochs | 20,069,187 | 79,614 |

`recipe.py` holds the decision loop and the evaluator; `common.py` the input features
and the decision head the MLP and GNN share.

## Inputs

* `--data-dir`: the gridfm-datakit export root of the Table 1 corpus, holding
  `case500_goc/raw/*.parquet`. Default `<data home>/case500_datakit/case500_30k`.
* `--pyg-dir`: its `.pyg.json` export, `train_opfdata/` (29,487 files) and
  `holdout_opfdata/` (500 files, `scenario_29500` ... `scenario_29999`). Default
  `<data home>/case500_datakit`. The baseline rows read their scenarios from these
  files, GridSFM through its own loader; the static grid comes from the parquet.
* graphkit rows only: `--graphkit-splits`, written by
  `graphkit.make_splits(raw_dir, out_dir)` (test = the hold-out, val = the last 10 % of
  the rest, train = the remainder), and `--graphkit-norm-stats`, the checkpoint
  `case500_graphkit_normalizer.pt`. Without it the normalizer is refitted on the train
  split, which moves baseMVA by 0.08 %.

## Commands (seed `s` = 0, 1, 2)

```
python -m powerform.training.case500 --arch mlp     --seed s
python -m powerform.training.case500 --arch gnn     --seed s
python -m powerform.training.case500 --arch gridsfm --seed s
python -m powerform.training.case500 --arch gridsfm --recipe state --seed s
# in graphkit's environment:
python -m powerform.training.case500 --arch graphkit --seed s --graphkit-splits SPLITS \
    --graphkit-norm-stats checkpoints/case500_graphkit_normalizer.pt
```

Each writes `outputs/case500/case500_<row>_seed<s>.{pt,json}`. Seed 0 of the GridSFM state
row runs as GridSFM's own seed 42.

`--eval-only` re-scores a checkpoint and reads the row from its config, so `--arch` is
not needed; the record goes to `outputs/case500/rescore/`:

```
python -m powerform.training.case500 --eval-only checkpoints/case500_mlp_seed0.pt
python -m powerform.training.case500 --eval-only checkpoints/case500_gridsfm_state_seed0.pt
python -m powerform.training.case500 --eval-only checkpoints/case500_graphkit_state_seed0
# in graphkit's environment:
python -m powerform.training.case500 --eval-only checkpoints/case500_graphkit_decision_seed0.pt \
    --graphkit-splits SPLITS --graphkit-norm-stats checkpoints/case500_graphkit_normalizer.pt
```

The first graphkit run on a corpus, even with `--limit 2`, reads all of its parquet into
memory and writes one file per scenario under `case500_goc/processed/` in the data home,
about 4.7 GB. Later runs reuse them.

`--limit N` scores the first `N` hold-out scenarios, with the values the full run gives
them. For that the GridSFM state row always runs the full run's inference batches of 25,
as its float32 forward moves at rounding level with the composition of the batch.

The graphkit state row is graphkit's own trainer, and its `predict` output directory
stands in for the checkpoint:

```python
from powerform.model.baselines.graphkit import (STATE_CONFIG, config_with_splits,
                                                launch_graphkit_cli)
cfg = config_with_splits(STATE_CONFIG, SPLITS, "outputs/state_80k.yaml")   # seeds 1, 2: edit `seed`
launch_graphkit_cli(["train", "--config", str(cfg), "--data_path", D,
                     "--exp_name", "graphkit_state", "--run_name", "s0",
                     "--log_dir", "outputs/mlruns"])
# once train has finished:
launch_graphkit_cli(["predict", "--config", str(cfg), "--data_path", D,
                     "--model_path", BEST_STATE_DICT,
                     "--normalizer_stats", NORMALIZER_STATS,
                     "--log_dir", "outputs/mlruns",
                     "--output_path", PRED_DIR])
```

Here `D` is `--data-dir`. `train` writes `BEST_STATE_DICT` and `NORMALIZER_STATS` as
`artifacts/model/best_model_state_dict.pt` and `artifacts/stats/normalizer_stats.pt`
under `outputs/mlruns/<experiment id>/<run id>/`. The configs select `accelerator: gpu`,
so `train` and `predict` need a CUDA GPU; re-scoring the dumps does not.

## Third-party code and licenses

* **GridSFM** (Microsoft, MIT): vendored under `powerform/third_party/gridsfm/` from commit
  `1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`, with its LICENSE and a NOTICE listing what is
  included. The module files are byte-identical to that commit (`tests/test_table1.py`). It
  needs `torch_geometric`, `scipy` and `huggingface_hub`.
* **gridfm-graphkit** (GridFM, Apache-2.0): not vendored. Install it in its own environment,
  with the versions these rows ran with. For a GPU (to re-score only, use `cpu` for `cu126`),
  from this repository's root:

  ```
  pip install -r powerform/model/baselines/graphkit_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 -f https://data.pyg.org/whl/torch-2.9.1+cu126.html
  git clone https://github.com/gridfm/gridfm-graphkit.git
  cd gridfm-graphkit && git checkout b3d663b62179222c1ebec00ee29f67ea50e68c0b
  pip install --no-deps -e .
  ```

  A plain `pip install -e .` takes today's latest packages, and with those the graphkit import
  fails. `pip check` still reports graphkit's own `torch>=2.10`; these rows ran on 2.9.1.

  `graphkit.py` holds the two rows; `graphkit_configs/` holds graphkit's published case500
  config with the changes this comparison makes, each documented in the file.

No released weights of either model are used: both are trained from random initialization
on the Table 1 corpus.
