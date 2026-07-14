# Attention Implementation Benchmark

RTX 5090 (Blackwell, CC 12.0), CUDA 13.2, PyTorch 2.10.0–2.12.0, Transformers 4.47.1–5.5.0, 32 GB VRAM

## Test Methodology
- 50-image COCO val2017 caption benchmark, `max_new_tokens=50`, `device_map="cuda"` (GPU only)
- Selection: **fastest inference FPS wins** (not load time)
- All configs tested in same hardware session with no other GPU load

## Direct-HuggingFace Models (in-process)

| Model | attn_implementation | Load (s) | 50-img Infer (ms) | FPS | Status | Chosen |
|-------|---------------------|---------:|-------------------:|----:|--------|--------|
| Florence-2-large-ft | flash_attention_2 | 4.2 | 211.8 | 4.72 | ✅ | |
| Florence-2-large-ft | **sdpa** | 3.6 | **209.6** | **4.77** | ✅ | **✓** |
| Florence-2-large-ft | eager | 3.6 | 207.7 | 4.81 | ✅ | |
| PaliGemma2-3B | flash_attention_2 | 6.8 | 306.9 | 3.26 | ✅ | |
| PaliGemma2-3B | **sdpa** | 6.0 | **214.1** | **4.67** | ✅ | **✓** |
| PaliGemma2-3B | eager | 5.9 | 235.8 | 4.24 | ✅ | |
| Llama-3.2-11B-Vision | flash_attention_2 | — | — | — | ❌ `'is_causal'` attr error | |
| Llama-3.2-11B-Vision | **sdpa** | 7.1 | **1845.3** | **0.54** | ✅ | **✓** |
| Llama-3.2-11B-Vision | eager | 7.2 | 2609.6 | 0.38 | ✅ | |
| Phi-3.5-vision-instruct | flash_attention_2 | — | — | — | ❌ unsupported | |
| Phi-3.5-vision-instruct | sdpa | — | — | — | ❌ unsupported | |
| Phi-3.5-vision-instruct | **eager** | 6.5 | **4151.0** | **0.24** | ✅ | **✓** |
| Cosmos-Reason1-7B | flash_attention_2 | 137.7¹ | 1287.8 | 0.78 | ✅ | |
| Cosmos-Reason1-7B | **sdpa** | 7.2 | **908.8** | **1.10** | ✅ | **✓** |
| Cosmos-Reason1-7B | eager | 6.9 | 1000.8 | 1.00 | ✅ | |
| Qwen3-VL-8B-Instruct | flash_attention_2 | 244.8² | 1658.0 | 0.60 | ✅ | |
| Qwen3-VL-8B-Instruct | **sdpa** | **3.3** | **1127.5** | **0.89** | ✅ | **✓** |
| Qwen3-VL-8B-Instruct | eager | 3.2 | 1238.8 | 0.81 | ✅ | |

¹ FA2 first-load includes CUDA kernel compilation for Cosmos (sm12.0); subsequent loads ~7s.
² FA2 first-load for Qwen3 includes Blackwell flash-attn kernel compilation (244s). Subsequent loads still ~100s due to flash-attn wheel import overhead.

## Subprocess Models

These models wrap external `run.py` scripts. Their own `attn_implementation` is set inside their loader.

| Model | attn_implementation | Load (s) | Infer (ms) | FPS | Status | Notes |
|-------|---------------------|---------:|-----------:|----:|--------|-------|
| Llava-v1.6-Mistral-7B | flash_attention_2 | — | — | — | ❌ ELF mismatch (wrong .so ABI) | symlinked .so from phi-4 venv incompatible |
| Llava-v1.6-Mistral-7B | **sdpa** | 6.5 | **1408.3** | **0.71** | ✅ | **chosen** |
| Llava-v1.6-Mistral-7B | eager | 6.6 | 1636.6 | 0.61 | ⚠️ OOM warnings | attempted ~1.0–1.1 GiB during generation |
| SigLIP2 | sdpa | — | — | — | ✅ | set in `siglip2/run.py` |
| MoonViT | sdpa | — | — | — | ✅ | set in `moonvit/run.py` |
| DINOv2 (dinov3) | sdpa | — | — | — | ✅ | set in `dinov3/run.py` |
| DINOtool | N/A | — | — | — | ✅ | uses `dinotool` package, not transformers |
| Phi-4-multimodal | flash_attention_2 | — | — | — | ✅ | set in `phi-4_multimodal/model_loader.py` |
| DiffusionGemma | sdpa | — | — | — | ✅ | subprocess-based |
| LocateAnything | sdpa | — | — | — | ✅ | subprocess-based |

## Qwen3 Results

| Model | attn_implementation | Load (s) | 50-img Infer (ms) | FPS | Status |
|-------|---------------------|---------:|-------------------:|----:|--------|
| Qwen3-VL-8B-Instruct | flash_attention_2 | 244.8¹ | 1658.0 | 0.60 | ✅ (built from source, py3.10/torch 2.12.0) |
| Qwen3-VL-8B-Instruct | **sdpa** | **3.3** | **1127.5** | **0.89** | ✅ **chosen** |
| Qwen3-VL-8B-Instruct | eager | 3.2 | 1238.8 | 0.81 | ✅ |
| Qwen3-VL-8B-Thinking | unsloth (default) | 151.5 | 2155.0 | 0.46 | ✅ (Unsloth, py3.11/torch 2.10.0) |

¹ Includes Blackwell flash-attn kernel compilation (244 s); subsequent loads still ~100 s due to flash-attn import.

## Final Configuration

### common.py (in-process HuggingFace models)

| Model | attn_implementation | Reason |
|-------|---------------------|--------|
| **Florence-2-large-ft** | `sdpa` | all within 2% (4.72–4.81 FPS); sdpa balance of speed/stability |
| **PaliGemma2-3B** | `sdpa` | sdpa: **4.67** > eager: 4.24 > FA2: 3.26 FPS |
| **Llama-3.2-11B-Vision** | `sdpa` | only robust option (FA2 broken, eager 0.38 FPS) |
| **Phi-3.5-vision-instruct** | `eager` | only supported option (FA2/sdpa unsupported by model) |
| **Cosmos-Reason1-7B** | `sdpa` | sdpa: **1.10** > eager: 1.00 > FA2: 0.78 FPS |
| **Qwen3-VL-8B-Instruct** | `sdpa` | sdpa: **0.89** > eager: 0.81 > FA2: 0.60 FPS; FA2 also 244s load |

### Subprocess model `run.py` files

| Model | File | attn_implementation |
|-------|------|---------------------|
| **Llava-v1.6-Mistral-7B** | `Llava/run.py` | `sdpa` (0.71 FPS vs eager 0.61; FA2: ELF mismatch) |
| **SigLIP2** | `siglip2/run.py` | `sdpa` |
| **MoonViT** | `moonvit/run.py` | `sdpa` |
| **DINOv2 (dinov3)** | `dinov3/run.py` | `sdpa` |
| **Phi-4-multimodal** | `phi-4_multimodal/model_loader.py` | `flash_attention_2` |
| **DiffusionGemma** | — | `sdpa` (subprocess) |
| **LocateAnything** | — | `sdpa` (subprocess) |

## OOM Details

| Model+Config | Attempted Alloc | VRAM Total | Error | Notes |
|:---|---|---|---|---|
| Llama-3.2-11B-Vision + 4-bit + eager (with `device_map="auto"`) | 2.47 GiB | 32 GiB | `CUDA OOM` | With `device_map="cuda"` eager worked (0.38 FPS) |
| Llava-v1.6-Mistral-7B + eager (mid-run, `device_map="cuda"`) | 0.97–1.04 GiB | 32 GiB | OOM warnings (non-fatal, ~990 MiB free) | sdpa avoids this entirely |
| Cosmos-Reason1-7B (any) | — | 32 GiB | None | — |

## Notes
- Florence-2 requires `device_map="cuda"` (not `"auto"`) because its custom code lacks `_no_split_modules`
- Phi-3.5-Vision config defaults to `flash_attention_2` but the model doesn't actually support it — explicitly set to `eager` via config override
- Llama-3.2-11B-Vision with 4-bit quantization: eager only works with `device_map="cuda"` (GPU-only); `device_map="auto"` triggers OOM from CPU->GPU offload overhead. sdpa is still recommended (0.54 FPS vs 0.38 FPS).
- Llava-v1.6-Mistral eager triggered repeated OOM warnings allocating ~1.0 GiB blocks when <1 GiB remained free
- flash-attn 2.7.4.post1 `.so` symlinked from phi-4_multimodal venv works for all other Python 3.13 models; Llava venv needs its own flash-attn build or the ELF/ABI mismatch persists
- Qwen3-instruct venv (py3.10, torch 2.12.0+cu130): flash-attn 2.8.3.post1 installed from cached cp310 wheel (fast). Qwen3-thinking venv (py3.11, torch 2.10.0+cu128): flash-attn built from source with `MAX_JOBS=4 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST="12.0"` (~98 min). Both verified working.
- Qwen3 thinking model uses Unsloth which handles its own attention internally; single config tested
