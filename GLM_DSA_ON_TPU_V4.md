# GLM-5 (GlmMoeDsaForCausalLM) on TPU V4 — MLA fallback mode

## Model tested
**ngxson/GLM-5-small-test** — a small GLM-5 test model with MLA + MoE + DSA architecture.

Served on TPU V4-8 (4 chips) by disabling the DSA indexer and falling back to
standard MLA attention, which tpu-inference supports.

## Config changes required
In the model's `config.json`:
```json
"index_topk": 0,
"num_nextn_predict_layers": 0
```
- `index_topk=0` disables DSA (sparse attention indexer), falling back to dense MLA
- `num_nextn_predict_layers=0` disables Multi-Token Prediction head

## vLLM patches required (vllm-project/vllm @ d860601)
In `vllm/model_executor/models/deepseek_v2.py`:
1. Change `self.is_v32 = hasattr(config, "index_topk")` to
   `self.is_v32 = bool(getattr(config, "index_topk", None))` (2 occurrences)
2. Skip indexer weights in `load_weights`: add
   `if "indexer" in name and not bool(getattr(self.config, "index_topk", None)): continue`
3. Fix `topk_indices_buffer` device: use `"cpu"` instead of `"tpu"` for torch.empty

## tpu-inference patches (this branch)
1. **TPU V4 RPA kernel support**: Add `case 4` to the RPA v3 kernel's TPU version match
2. **Mamba page size fix**: Recalculate mamba_page_size_padded for UniProcExecutor TP mismatch
3. **mrope API compat**: Fallback for changed get_mrope_input_positions signature
4. **GMM VMEM fix**: Reduce vmem_limit factor from 0.9 to 0.7 for V4's 16MB VMEM
5. **GMM tile size cap**: Cap tile sizes for V4's smaller VMEM
6. **RPA default fix**: Use `max(1, 512 // page_size)` to avoid bkv_p=0
7. **Tuned RPA data**: Added `tpu_v4.json` registry for RPA block sizes
