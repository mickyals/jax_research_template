# cyclone_jax

Tropical-cyclone intensity classification from sparse in-situ observations
(LISO land / MISO marine / CUON upper-air, IBTrACS driver), built on the
arcana volume/bookshelf data layer. Branch of record: `jrtv2`.

## Layout

```
cyclone_jax/
├── config.py        # load_config: resolves train-config pointers, validates keys
├── configs/
│   ├── data/        #   scenarios: overfit / train / memorise / test / multistorm
│   ├── models/      #   one yaml per model: mlp / siren / finer (+ wandb tags)
│   └── train/       #   entry points: train.yaml points at {data, model}
├── data/            # the whole data side — see data/usage_doc.md
├── models/          # MODELS registry + build_model — see models/usage_doc.md
├── train/           # train.py entry + losses/metrics/log builders
│                    #   (tune.py / evaluate.py: plan steps 6-7)
├── visualise/       # figures.py: storm panel/sequence/gif mechanics
├── notebooks/       # run_experiment.ipynb — cells ready (kernel 'jrt')
└── runs/            # run artifacts, gitignored (**/runs/)
```

## Quick start (notebook)

```python
from experiments.cyclone_jax.config import load_config, CONFIG_DIR
from experiments.cyclone_jax.data.interface import build_data

cfg  = load_config(CONFIG_DIR / 'train' / 'train.yaml')
data = build_data(cfg['data'], seed=cfg['trainer']['seed'])

sample = data.loader.build(0)          # one named ragged sample {'x', 'y'}
batch  = next(iter(data.streams['train']))   # device-ready batch

from experiments.cyclone_jax.models import build_model
model, tags = build_model(cfg['model'], data.targets)   # tags -> wandb
```

## Training (CLI)

```bash
export PYTHONPATH=jrt
export WANDB_API_KEY=...           # trainer.logger: wandb
python -m experiments.cyclone_jax.train.train \
    jrt/experiments/cyclone_jax/configs/train/train.yaml --gpu 0
```

- Model swap = edit the entry yaml's `model:` pointer (`mlp` / `siren` /
  `finer`); run name defaults to `{model}-{data}-s{seed}`, run tags =
  model tags + data tags.
- `--gpu N` (or top-level `gpu:` in the entry yaml) pins
  `CUDA_VISIBLE_DEVICES` on multi-GPU boxes; a shell setting always wins.
- Multi-CPU boxes (Linux): set `num_workers` in the DATA yaml —
  multiprocess sample assembly for the train stream (`training/prefetch`).
- Each run writes `run_dir/norm_stats.json` (evaluation must reuse it) and
  `run_dir/data_manifest.json` (split sizes/class counts + merged config);
  figures land in `run_dir/figures/` and on the logger (confusion matrices,
  storm panels per `trainer.callbacks`; end-of-run test CM + storm
  sequence gif per the data yaml's `storm_panels` block).
- Full-dataset memorisation probe: `data: memorise`; run
  `data.identifiability.input_collisions` first for the accuracy ceiling.
- Dataset variants: `train` / `memorise` (land+marine) have `_land` /
  `_marine` clones — swap the entry yaml's `data:` pointer. All variants
  share `pad_to: 1536` on purpose: the MLP input width is `pad_to × F`
  (`features.flatten`), so a shared pad keeps architectures identical
  across variants and the jit shape single.

## Hyperparameter search (CLI)

```bash
python -m experiments.cyclone_jax.train.tune \
    jrt/experiments/cyclone_jax/configs/train/tune_memorise_mlp.yaml --gpu 0
```

- The tune yaml points at a BASE train config and maps DOTTED paths into
  the merged `{data, model, trainer}` config to search specs
  (`{low, high, log?}` / `{choices: [...]}`) — architecture, data and
  optimisation HPs through one mechanism.
- Study direction derives from the base trainer's `patience_direction`
  (the objective IS the patience metric) — there is no direction key.
- Record (no sqlite): `<run_dir>/<study>/trials.csv` appended after every
  trial + one wandb run per trial (group = study, run `{study}-t{N}`)
  + `best.yaml` — the merged winning config, self-contained.
- `retrain_best: true` retrains the winner under `<study>/best` with the
  full end-of-run records, wandb-tagged `{study}-best`.

## Linux transfer (multi-CPU/GPU box)

The configs are machine-agnostic; the shell names the site. Nothing in
the code path is Windows-specific — the Windows-side caveats (workers,
OneDrive/orbax) simply disappear on Linux.

```bash
# Environment — requirements.txt is the single source of truth:
#   conda box:     conda env create -f environment.yaml && conda activate jrt
#   pip-only box (venv is all the cluster gives you):
python3 -m venv ~/jrt-venv
. ~/jrt-venv/bin/activate
pip install -r requirements.txt            # jax[cuda13] default; wandb included
python -m ipykernel install --user --name jrt --display-name "Python (jrt)"

export PYTHONPATH=jrt
export WANDB_API_KEY=...

# site overrides — beat the yamls' root/num_workers (machine properties;
# the yamls never need editing between boxes):
export CYCLONE_JAX_ROOT=/data/Caribbean-Obs
export CYCLONE_JAX_NUM_WORKERS=8            # ~CPU cores; fork = Linux payoff

python -m experiments.cyclone_jax.train.train \
    jrt/experiments/cyclone_jax/configs/train/train.yaml --gpu 0
```

- Copy the built library (volumes + bookshelf + `meta.json` sidecars) to
  local disk — it is read via mmap, so network filesystems hurt.
- `--gpu N` pins one device per run; run one process per GPU with
  different `--gpu` values for parallel runs (a shell
  `CUDA_VISIBLE_DEVICES` always wins over `--gpu`/yaml).
- `num_workers > 0` uses fork workers that inherit the mmaps for free
  (the reason it must stay 0 on Windows/spawn).
- From a notebook: `notebooks/run_experiment.ipynb` has the cells ready
  (kernel `jrt`). Rules it encodes: `CUDA_VISIBLE_DEVICES` + `sys.path`
  in the FIRST cell (JAX initialises once per kernel — restart to switch
  GPU); fresh `run_dir` per run (OneDrive/orbax gotcha on Windows);
  restart the kernel between big runs (device memory accumulates).

## Principles

- **One seed** (`trainer.seed`) populates jax model init
  (`utils.jax_core.helpers.create_rng`) and numpy data order (Sampler).
- **Data modules are jax-free** (multiprocess-worker purity); jit sees one
  static shape via a fixed `pad_to` at collate.
- **Named dicts everywhere** — no positional token matrices in the data
  layer; models decide their own packing/encoding.
- **Leakage allowlist**: model input = obs channels + relative time +
  position only; IBTrACS intensity/structure columns (`CYC_TARGETS`) are
  target/metadata, never features.
- **Canonical SI storage** (m, Pa, m/s; catalogue + meta.json sidecars +
  display-unit helper in `data/variables.py`); IBTrACS converted at build,
  after the kt-threshold SSHS remap. Details: `data/usage_doc.md` Units.
- Experiments import `jrt/*` freely; `jrt` never imports experiments.

## Tests

```bash
/c/Users/micke/anaconda3/envs/jax-research-template/python.exe -m pytest \
    tests/experiments/cyclone_jax/ tests/datasets/ -q
```
