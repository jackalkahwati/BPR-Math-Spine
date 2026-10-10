# Synthetic fixture for scripts/check_openai_math_pin.py

Everything under `upstream/` is a minimal stand-in written for BPR's tests. None
of it is copied from openai/math: the families (901, 902), titles, statements,
comparator files and licence text are invented. `index.json` is a synthetic
index in the same schema as `doc/external/openai_math_index.json`, with sha256
values computed from these stand-ins.

The tests copy `upstream/` to a temporary directory before running the checker,
because the checker refuses to read a clone inside the BPR tree.
