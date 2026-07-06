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
