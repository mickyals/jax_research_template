"""
experiments/sparse_obs_encoder/data/transforms/

Swappable input transforms, selected declaratively by an InputSpec
(data/inputs.py):

    normalise.py  — observation normalisers (minmax_01/minmax_11/standardise)
    encoding.py   — coordinate encoders/decoders (unit_circle/domain)
    derived.py    — derived obs variables (wind components)

Each is a small ``utils.registry.Registry`` so new transforms register by name
without touching the dataset assembly. Missingness handling is NOT a transform
— it is invariant and stays structural in dataset.py.
"""
