# Running tpu-inference on a v4-16 TPU pod (bare metal, no Docker)

Tested on a v4-16 (2 hosts, 8 JAX devices in megacore mode) running Ubuntu 20.04.

## Prerequisites

- A v4-16 TPU pod with SSH access between hosts
- `multihost_dev.sh` copied to `/home/mark/multihost_dev.sh` on host 0
- Source it: `source /home/mark/multihost_dev.sh`
- Verify connectivity: `status`

## 1. Generate SSH key (host 0 to host 1)

```bash
# On host 0:
gcloud alpha compute tpus tpu-vm ssh $(echo $TPU_NAME) \
    --zone=$TPU_ZONE --project=$TPU_PROJECT --worker=1 --command="hostname"
```

This auto-generates `~/.ssh/google_compute_engine` on first run. Verify:

```bash
on 1 hostname
```

## 2. Install Python 3.11 via miniconda

```bash
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
$HOME/miniconda3/bin/conda create -y -n tpu python=3.11
```

## 3. Install system deps

```bash
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
```

## 4. Clone repos and install from source

```bash
export PATH="$HOME/miniconda3/envs/tpu/bin:$PATH"
pip install uv

# Clone vLLM at the pinned commit
VLLM_COMMIT=$(cat /home/mark/tpu-inference/.buildkite/vllm_lkg.version)
git clone https://github.com/vllm-project/vllm.git /home/mark/vllm
cd /home/mark/vllm && git checkout "$VLLM_COMMIT"

# Install vLLM TPU deps + vLLM
uv pip install --system -r requirements/tpu.txt
VLLM_TARGET_DEVICE="tpu" uv pip install --system -e .

# Install tpu-inference from source (over the pip version)
cd /home/mark/tpu-inference
uv pip install --system -e .
```

## 5. Sync everything to workers

```bash
source /home/mark/multihost_dev.sh
sync-files
```

Verify:

```bash
on 1 "$HOME/miniconda3/envs/tpu/bin/python3 --version"
# Should print: Python 3.11.x
```

## 6. Start Ray cluster

```bash
ray-up
```

Wait a few seconds, then confirm both nodes are connected:

```bash
$HOME/miniconda3/envs/tpu/bin/ray status
```

You should see `8.0 TPU` and 2 active nodes.

## 7. Launch the model

```bash
TPU_MULTIHOST_BACKEND=ray \
$HOME/miniconda3/envs/tpu/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-4B" \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --distributed-executor-backend ray \
    --port 8000
```

Wait for model download + compilation (~2 min). The server is ready when you see HTTP request logs.

## 8. Test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}' \
  | python3 -m json.tool
```

## Important notes

### v4-16 = 8 devices, not 16

v4 runs in **megacore mode** (2 cores fused per chip = 1 JAX device per chip).
v4-16 = 8 chips total = **8 JAX devices**. Use `--tensor-parallel-size 8`, not 16.

### Troubleshooting

If anything gets stuck or the TPU is locked by a stale process:

```bash
nuke
```

This kills all python/ray/TPU processes on all hosts and cleans up temp state.

Then restart from step 6 (ray-up).

### What the patches fix

This branch includes 3 fixes for bare-metal v4 multi-host:

1. **Lazy imports in `ray_distributed_executor.py`**: JAX was importing at module load time (when the Ray actor class is deserialized), before multi-host env vars were set. Moved `tpu_inference` imports from top-level to inside methods so env vars take effect first.

2. **Multi-host env var injection**: For PP=1 multi-node, the executor now injects `TPU_PROCESS_ADDRESSES`, `CLOUD_TPU_TASK_ID`, and `TPU_PROCESS_BOUNDS` into worker env vars so JAX coordinates across hosts and sees all 8 devices.

3. **v4 attention kernel support**: The RPA v3 kernel only had tuned block sizes for v5/v6/v7. Added `case 4` using v5/v6 defaults.

4. **Multi-host memory stats**: `hbm_usage_bytes()` now checks `jax.process_count() > 1` in addition to `TPU_MULTIHOST_BACKEND == "ray"` to correctly handle non-addressable remote devices.
