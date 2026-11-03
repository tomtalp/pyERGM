# Test data

`hardly_reproducible_gt_model.npz`: the ground truth (GT) and 1024 networks sampled from it for `TestHardlyReproducibleGTModel` in `tests/test_mple_optimization.py`, which documents the model's features. sha256: `a06d001079e57b3aba53a27cd1d1267505e7603885a633f113c4e71dab2d3026`. Arrays: `gt_thetas` (127,); `types_10` and `types_5` (100,) int; `positions` (100, 3); `node_attributes` (100,); `networks_packed`, uint8 bit-packed (`np.unpackbits`); `networks_shape` = `[100, 100, 1024]`.
