"""Model-Based First-Person Shooter Agent.

Setting PYTORCH_ENABLE_MPS_FALLBACK at the package root is what makes the
ordering guarantee project-wide. Python runs a parent package's `__init__`
before any submodule body, so this precedes every `import torch` in the
codebase -- including modules like `mbfps.data.features` that import torch at
the top of their own file, before importing anything from mbfps.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
