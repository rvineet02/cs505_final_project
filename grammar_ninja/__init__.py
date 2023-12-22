import os
HF_HOME = os.environ.get("HF_HOME", None)
assert HF_HOME is not None, "HF_HOME is not set"