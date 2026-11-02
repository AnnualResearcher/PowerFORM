# Getting the data

Every dataset the paper uses, how to obtain it, and how to check that what you
have is what the paper's runs read. All commands run from the repository root.

## Where the data lives: `POWERFORM_DATA`

Every corpus sits under one directory, the **data home**: `$POWERFORM_DATA` if
set, otherwise `<repo>/datasets`. The fetch and convert scripts write there and
the loaders read from there, both taking the location from `powerform/paths.py`.

```
export POWERFORM_DATA=/data/powerform          # bash
$env:POWERFORM_DATA = "D:\powerform"           # PowerShell
```

Set it in the environment before the process starts: it is read once, at import.
Under the data home:

| subdirectory | contents | written by |
|---|---|---|
| `opfdata_hf/<pglib case>/<variant>/` | OPFData, one gzipped JSONL shard per split + `manifest.json` | `fetch_opfdata.py` |
| `opfdata_case2000/<variant>/`, `opfdata_case6470/<variant>/` | OPFData test split, `example_14250.json.gz` .. `example_14999.json.gz` | `fetch_test_split.py` |
| `case500_datakit/case500_30k/case500_goc/raw/` | gridfm-datakit output (parquet) | `gridfm_datakit generate` |
| `case500_datakit/train_opfdata/`, `case500_datakit/holdout_opfdata/` | that corpus, one OPFData-shaped `scenario_<id>.pyg.json` per scenario | `datakit_export.py` |

## What the paper uses

| result | dataset | split read | command |
|---|---|---|---|
| Table 2 (OOD) training, App. B | OPFData `case57`, `case118`, `case500_goc`, `fulltop` | train (8,000 of 13,500 per grid) | step 1 |
| Table 2, training-grid columns | same grids, `fulltop` and `n1` | test (examples 14250-14999) | step 1 |
| Table 2, `case30`* | OPFData `case30_ieee`, `fulltop` | test | step 1 |
| Table 2, `case2000`*, `case6470`* | OPFData `case2000_goc`, `case6470_rte`, `fulltop` | test | step 2 |
| Table 1 (ID), App. B | gridfm-datakit `case500_goc`, 30k scenarios | train = ids < 29,500 (29,487 solved); hold-out = ids 29,500-29,999 | step 4 |
| Section 4.2 (reconfiguration) | OPFData `case57`, `fulltop` | train and val | step 1 |

## 1. OPFData shards (training grids and `case30`)

```
python -m powerform.data.corpus.fetch_opfdata --all --variant both
```

Streams each `<case>_0.tar.gz` from `https://storage.googleapis.com/gridopt-dataset`
(`dataset_release_1` for `fulltop`, `dataset_release_1_nminusone` for `n1`) and
writes three shards per (case, variant). `--all` is `case30`, `case57`, `case118`,
`case500_goc`; `--case pglib_opf_case57_ieee` fetches one. The paper needs
`case57`/`case118`/`case500_goc` in both variants and `case30` `fulltop`;
`--all --variant both` also fetches `case30` `n1` (40 MB). A finished (case,
variant) is skipped on a re-run. `--dry-run` prints the URLs and destinations
without fetching.

| case | per variant on disk | fetch time |
|---|---|---|
| `pglib_opf_case30_ieee` | 40 MB | 0.3 min |
| `pglib_opf_case57_ieee` | 78 MB | 0.5 min |
| `pglib_opf_case118_ieee` | 235 MB | 1.5 min |
| `pglib_opf_case500_goc` | 975 MB | 6-7 min |

Total for `--all --variant both`: 2.66 GB. Every manifest lists 13,500 / 750 / 750
examples (train / val / test), split by PyG's `OPFDataset(num_groups=1)` rule:
train `i < 13500`, val `i < 14250`, test `i >= 14250`.

## 2. OPFData test splits (`case2000`, `case6470`)

```
python -m powerform.data.corpus.fetch_test_split --case case2000 --variant fulltop
python -m powerform.data.corpus.fetch_test_split --case case6470 --variant fulltop
```

Only the 750 test members are kept, since the zero-shot grids are never trained
on, but the archive is streamed until the last of them has passed: 4-5 min per variant
at ~18 MB/s. On disk:
223 MB per variant for `case2000`, 643 MB for `case6470`. `--variant both` adds
the `n1` split, which the paper's tables do not use.

## 3. Verify what you have (offline)

```
python -m powerform.data.corpus.fetch_opfdata --all --variant both --verify-only --deep
python -m powerform.data.corpus.fetch_test_split --case case2000 --variant fulltop --verify-only --deep
python -m powerform.data.corpus.fetch_test_split --case case6470 --variant fulltop --verify-only --deep
```

`--verify-only` never opens a network connection. Without `--deep` it checks the
manifest (case, variant, source URL, split rule, per-split counts and index
ranges, the three splits covering 0..14999 once) and the files' presence. `--deep`
also decompresses everything and compares line counts, byte counts and content
SHA-256 with [`checksums.json`](checksums.json) (~90 s for all of the above). The
digests are over the decompressed content: gzip records a timestamp, so the `.gz`
bytes differ on every fetch while the content does not. A fresh fetch matches
every digest unless OPFData re-publishes the archives.

Expected content digests (full values in `checksums.json`):

| dataset | split | count | SHA-256 (first 12 digits) |
|---|---|---|---|
| `case57` `fulltop` | train / val / test | 13,500 / 750 / 750 | `49e2fa8fe101...` / `9897f42aa489...` / `302b35675f7f...` |
| `case118` `fulltop` | train / val / test | 13,500 / 750 / 750 | `b2eea575be47...` / `9dd0d78864b8...` / `38a9b9625386...` |
| `case500_goc` `fulltop` | train / val / test | 13,500 / 750 / 750 | `18cc57d52017...` / `3b24021ecc54...` / `b91b0e837082...` |
| `case30` `fulltop` | test | 750 | `1cd81ab91557...` |
| `case57` / `case118` / `case500_goc` `n1` | test | 750 each | `749ae3cc9b57...` / `e616db0b478a...` / `af9d5f45d4fa...` |
| `case2000` `fulltop` | test | 750 files | `b93ca531babb...` |
| `case6470` `fulltop` | test | 750 files | `5c8dc5e769b3...` |

## 4. The Table 1 corpus (gridfm-datakit)

The in-distribution corpus is generated, not downloaded: 30,000 `case500_goc` load
scenarios, each solved by AC-OPF (IPOPT through PowerModels.jl), with the
configuration in [`datakit/config_case500_30k.yaml`](datakit/config_case500_30k.yaml)
(seed 0; aggregate load profile scaled into [l, u] with per-load uniform noise
sigma = 0.2; no topology, generation or admittance perturbation).

**Tool.** [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) (Apache-2.0),
version 1.0.5 at commit `2d21c6855d84987832b11eb6a651b9ec98dde47b`. It is not
vendored here. It needs Python 3.10-3.12 and installs its own Julia, so install it
in a separate environment:

```
python -m pip install "git+https://github.com/gridfm/gridfm-datakit@2d21c6855d84987832b11eb6a651b9ec98dde47b"
```

The Julia side resolves on first use from the package's `juliapkg.json`. The paper's
corpus was generated with Julia 1.11.9, PowerModels.jl 0.21.6, Ipopt.jl 1.15.0
(Ipopt_jll 300.1400.1902, i.e. Ipopt 3.14.19), JuMP 1.31.1, MathOptInterface 1.51.2,
juliacall 0.9.35, and on the Python side numpy 2.4.6, pandas 3.0.5, pyarrow 25.0.0.
datakit downloads the case file from `power-grid-lib/pglib-opf` (`master`) on first
use; the one used is the v23.07 release of `pglib_opf_case500_goc.m` (CC BY 4.0),
SHA-256 `36c298d571605019ef16c17dd74680adca1386d91ed47d69a0d909aebc90a1b6`
(212,013 bytes; in datakit's package directory `gridfm_datakit/grids/`).

**Generate** (from the repository root, in that environment; single process, 11 h
38 min on the machine that produced the paper's corpus):

```
gridfm_datakit generate powerform/data/corpus/datakit/config_case500_30k.yaml
```

Its `settings.data_dir` is `./datasets/case500_datakit/case500_30k`, the default
data home seen from the repository root; if `POWERFORM_DATA` is set, change it in a
copy of the config to `<POWERFORM_DATA>/case500_datakit/case500_30k`, and nothing
else.

What the run must leave in `raw/` (`args.log`, `scenarios_agg_load_profile.log`,
`error.log`, `n_scenarios.txt`):

- `u=1.2000000000000004`, `l=0.7200000000000002` (the aggregate band: `u` from a
  short OPF feasibility search in steps of 0.05 from 0.8, `l = 0.6 u`);
- 29,987 solved scenarios; 13 OPF failures, all at ids 16496-16582 (16496,
  16498-16507, 16509, 16582: 12 `LOCALLY_INFEASIBLE`, 1 `ITERATION_LIMIT`);
- 1.8 GB of parquet. The graphkit rows add a `processed/` directory of about 4.7 GB
  on first use; nothing else reads it.

The sampled load profile precedes all solves, so it can be checked after a few
minutes: `datakit_export.raw_summary()` hashes it, and the expected digest is
`60ea614bb0d718e8...` (`checksums.json`). The IPOPT solves themselves are not
guaranteed bit-reproducible across machines.

**Convert** to one OPFData-shaped file per scenario (needs numpy, pandas, pyarrow;
reads the OPFData `case500_goc` train shard from step 1 as the structural
reference):

```
python -m powerform.data.corpus.datakit_export --split holdout
python -m powerform.data.corpus.datakit_export --split train
python -m powerform.data.corpus.datakit_export --split holdout --verify-only --deep
python -m powerform.data.corpus.datakit_export --split train --verify-only --deep
```

| split | ids | files | on disk | content SHA-256 (first 12 digits) |
|---|---|---|---|---|
| hold-out | 29,500-29,999 | 500 | 124 MB | `020a8e3aa54d...` |
| train | 0-29,499 minus the 13 failures | 29,487 | 7.3 GB | `0686a28d4d18...` |

Before writing, the converter asserts that every scenario-invariant field it builds
from the parquet equals the OPFData reference scenario (bus, generator, shunt, edge
features and incidence, link tables), and it verifies the branch-flow label column
order on the first scenario. The digests match only a conversion of the paper's raw
corpus; a regenerated one is checked by counts and by step 5.

## 5. Check the operating band (App. B)

```
python -m powerform.data.corpus.load_factors
```

prints the load factors (scenario total demand / nominal, nominal = median total
demand of 400 OPFData `case500_goc` train scenarios = 17,768.8 MW). For the paper's
corpus:

| source | scenarios | factor range |
|---|---|---|
| sampled profile, all | 30,000 | 0.711-1.204 |
| sampled profile, hold-out | 500 | 0.840-1.052 |
| solved corpus, all | 29,987 | 0.711-1.188 |
| converted hold-out | 500 | 0.840-1.052 |

The realised range extends slightly beyond [0.72, 1.20] because the per-load noise
moves each scenario's total. The 11 highest-demand scenarios of the profile are all
among the 13 failed solves, so the solved corpus tops out at 1.188; App. B's
0.711-1.204 is the generated profile.

## Third-party data and tools

| what | license | role |
|---|---|---|
| OPFData (Lovett et al., 2024), `gs://gridopt-dataset` | CC BY 4.0, (c) DeepMind (the bucket's `LICENSE`) | downloaded, not redistributed |
| PGLib-OPF v23.07 `case500_goc` | CC BY 4.0 (file header) | fetched by gridfm-datakit |
| gridfm-datakit 1.0.5 | Apache-2.0 | installed separately, not vendored |
| PowerModels.jl 0.21.6 | BSD-style (LANL) | installed by gridfm-datakit |
| Ipopt.jl 1.15.0 / Ipopt 3.14.19 | MIT / EPL-2.0 | installed by gridfm-datakit |

## Tests

```
python powerform/tests/test_data.py
```

checks the loaders, the converter, the offline verification and fetch URLs and the
`POWERFORM_DATA` override, skipping with the way to get it what is not on disk. Pass
`--deep` for the App. B numbers, which read two columns of the full bus table.
