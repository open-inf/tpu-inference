# Running GLM-5 (GlmMoeDsaForCausalLM) on TPU V4

This guide covers running GLM-5 models on TPU V4 hardware by disabling DSA
(DeepSeek Sparse Attention) and falling back to standard MLA (Multi-head
Latent Attention).

## Background

GLM-5 uses the `GlmMoeDsaForCausalLM` architecture, which combines:
- **MLA** (Multi-head Latent Attention) for KV compression
- **DSA** (sparse attention indexer with FP8 KV cache) for long-context efficiency
- **MoE** (Mixture of Experts)

DSA requires FP8 hardware support (TPU v5e+). On TPU V4, we disable DSA and
use dense MLA attention instead. This works because DSA is an optimization on
top of MLA — the underlying attention mechanism is identical.

**Model tested:** `ngxson/GLM-5-small-test`

## Setup

### 1. Clone and install

```bash
# Clone this branch (includes TPU V4 fixes)
git clone -b glmdsa https://github.com/open-inf/tpu-inference.git
cd tpu-inference

# Clone vLLM at the pinned commit and apply the DSA patch
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version)"
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "${VLLM_COMMIT_HASH}"
git apply ../patches/vllm-glmdsa-tpuv4.patch
cd ..

# System deps
sudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev

# Python environment
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv vllm_env --python 3.12
source vllm_env/bin/activate

# Install vLLM for TPU
cd vllm
uv pip install -r requirements/tpu.txt
VLLM_TARGET_DEVICE="tpu" uv pip install -e .
cd ..

# Install tpu-inference
cd tpu-inference
uv pip install -e .
cd ..
```

### 2. Patch the model config

Download the model and set `index_topk=0` to disable DSA:

```python
from huggingface_hub import snapshot_download
import json, os

path = snapshot_download("ngxson/GLM-5-small-test", cache_dir="/dev/shm")
config_path = os.path.join(path, "config.json")

with open(config_path) as f:
    config = json.load(f)

config["index_topk"] = 0                    # disable DSA -> use MLA
config["num_nextn_predict_layers"] = 0       # disable MTP head

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Patched: {config_path}")
```

### 3. Serve

```bash
export MODEL_IMPL_TYPE=vllm
vllm serve /dev/shm/models--ngxson--GLM-5-small-test/snapshots/<hash> \
    --max-num-seqs 16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --tensor-parallel-size 1
```

## What this branch changes (vs upstream tpu-inference)

| File | Change |
|------|--------|
| `kernels/ragged_paged_attention/v3/kernel.py` | Add TPU V4 support to RPA kernel version match |
| `kernels/ragged_paged_attention/v3/tuned_block_sizes.py` | Fix `bkv_p=0` when page_size > 512 on V4 |
| `kernels/megablox/gmm_v2.py` | Reduce VMEM limit factor for V4's 16MB VMEM |
| `kernels/megablox/tuned_block_sizes.py` | Cap GMM tile sizes on V4 |
| `platforms/tpu_platform.py` | Fix Mamba page size for UniProcExecutor TP mismatch |
| `runner/persistent_batch_manager.py` | Compat fallback for mrope API change |
| `kernels/tuned_data/ragged_paged_attention/v3/tpu_v4.json` | Tuned RPA block sizes for V4 |
| `patches/vllm-glmdsa-tpuv4.patch` | vLLM patch: disable DSA, fix device, skip indexer weights |

## Limitations

- **No DSA**: Sparse attention is disabled; the model uses dense MLA over full context. This increases attention compute for long sequences but is functionally correct.
- **No FP8**: TPU V4 lacks FP8 MXU support. Models must run in BF16.
- **TPU V4 is unsupported upstream**: These patches are specific to V4's 16MB VMEM and missing kernel support. They are not needed on v5e/v6e/v7.
