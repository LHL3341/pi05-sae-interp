#!/usr/bin/env bash
# Apply openpi's transformers_replace patch into the active venv.
# Required: pi0_pytorch.py asserts transformers==4.53.2 + patched siglip/gemma/paligemma.
set -euo pipefail

cd "$(dirname "$0")"
SRC=openpi/src/openpi/models_pytorch/transformers_replace
DST=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

echo "patching $DST"
cp -rv "$SRC"/* "$DST/"

python -c "
from transformers.models.siglip import check
ok = check.check_whether_transformers_replace_is_installed_correctly()
print('patch verification:', ok)
assert ok
"
