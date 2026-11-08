#!/bin/bash
# One-off environment setup on the SoC compute cluster, following the course's SoC cluster quick-start guide (Canvas).
#
# Run this INSIDE an interactive compute-node shell (never on xlogin, which has too little memory for conda):
#
#   salloc -t 120            # reserve a node for two hours (add -G h100-47 if you also want to test the GPU)
#   srun --pty bash          # shell on the reserved node
#   cd ~/cs4248-project && bash scripts/setup_env_cluster.sh
#   exit; exit               # leave the node and release the allocation
#
# It installs Miniconda if missing, creates the conda env ($CONDA_ENV, default cs4248, Python 3.11), installs the
# requirements with pip, downloads the external evaluation sets and all pre-trained weights (so GPU jobs do not
# depend on the network), and runs the unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cs4248}"

if [ -n "${SLURM_JOB_ID:-}" ] || [[ "$(hostname)" != xlogin* ]]; then
  echo "running on $(hostname)"
else
  echo "ERROR: run this on a compute node (salloc -t 120 && srun --pty bash), not on xlogin" >&2
  exit 1
fi

if [ ! -d "$CONDA_BASE" ]; then
  echo "== installing Miniconda to $CONDA_BASE =="
  curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_BASE"
  rm -f Miniconda3-latest-Linux-x86_64.sh
  "$CONDA_BASE/bin/conda" init bash
fi
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^$CONDA_ENV\s"; then
  echo "== creating conda env $CONDA_ENV =="
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"
set -u

echo "== installing requirements =="
pip install --upgrade pip
# The cluster's NVIDIA driver is CUDA 12.x (12.9 as of Nov 2026); the default PyPI wheel is built for CUDA 13 and
# cannot see the GPU, so install PyTorch from the matching CUDA 12 index first (override with TORCH_CUDA=cu126 etc.).
TORCH_CUDA="${TORCH_CUDA:-cu128}"
pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install -r requirements.txt

echo "== external evaluation sets =="
mkdir -p runs weights logs/slurm data/paraphrases
python download_eval_sets.py

echo "== pre-downloading model weights =="
python - <<'EOF'
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
for m in ["bert-base-uncased", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", "google/bert_uncased_L-2_H-128_A-2"]:
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
for m in ["Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"]:
    MarianTokenizer.from_pretrained(m); MarianMTModel.from_pretrained(m)
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("models cached")
EOF

echo "== unit tests =="
python tests/test_core.py
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: PyTorch cannot see the GPU. If the message above says the driver is too old, re-run with a lower CUDA build:" >&2
  echo "       TORCH_CUDA=cu126 bash scripts/setup_env_cluster.sh" >&2
  exit 1
}
echo "setup done on $(hostname) at $(date); submit jobs from xlogin with: sbatch scripts/job.sbatch"
