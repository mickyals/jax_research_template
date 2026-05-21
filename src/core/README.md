# Core

Reusable neural building blocks.

## Files

- `layers.py` - Attention, cross-attention, set aggregation, feed-forward blocks, layer norm, type-specific projection layers, masked token padding layer
- `activations.py` - Standard activations (ReLU, GELU, SiLU), SIREN sine activation for INR, softplus for variance output head
- `embeddings.py` - Cyclic time encoding, fourier features, spatial positional encodings, delta t encodings, learned vs fixed variants
- `nets.py` - Encoder, probabilistic field network, hypernetwork variant, full model
