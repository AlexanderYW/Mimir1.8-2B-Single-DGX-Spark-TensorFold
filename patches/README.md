# Patches

Applied in order with `git apply` on TensorFold [v1.3.7](https://github.com/ashhart/TensorFold/releases/tag/v1.3.7)
(`b3366549`) when `scripts/prepare.sh` builds the image. All three apply cleanly to a fresh v1.3.7 checkout
(`git apply --check`), and every touched Zig file passes `zig ast-check` and `zig fmt --check` with Zig 0.17.0. The
patched tree builds (`fatbins native`, `-Doptimize=fast`), its host tests pass (`test-cuda-dense-host`,
`test-hrm-text-cuda-host`, `test-dots-mocr-cuda-host`, `check-native-cuda-host`), and both BF16 and NVFP4 served
forwards are verified against transformers (see the main README's "Checking it").

Stock v1.3.7 has no `hrm_text` family, so `tensorfold-native serve` refuses Mimir v1.8 with "the native CUDA engine has
no backend for hrm_text checkpoints yet". HRM-Text is close to a dense decoder, which TensorFold's shared CUDA dense
engine (`zig/src/cuda_dense`, the one dots.mocr runs on) already serves: paged KV, lanes, draws, context-copy drafts and
the OpenAI server. The first two patches add what it lacks and the family on top; the third adds DGX Spark (GB10)
performance optimizations informed by our Qwen3.8 Flash Next TensorFold work.

## 0001-cuda-dense-prefix-lm-mha-prompt-rows.patch

Changes the shared dense engine and one kernel file. Existing recipes (dots.mocr) keep their geometry and behaviour,
since every new field has a default that keeps the old behaviour.

- **PrefixLM prompts** (`Geometry.prefix_lm`). Mimir attends bidirectionally over the whole prompt (transformers:
  `token_type_ids == 1` is one bidirectional block; vLLM and the release's own pipeline mark the whole rendered prompt)
  and causally over the reply. `prefill_attention.cu` gets `pattn_prefix_kernel`: `pattn_kernel`'s paged body with
  `causal = false`, so each row of a chunk sees keys `[0, p0 + W)`. `attentionShape` picks it for a PrefixLM geometry,
  and refuses by name (`UnsupportedDensePrefix`) where no instance is built. It never falls back to causal attention.
  Decode and verify rows keep the causal paged kernels, as transformers' generate does after its first step.
- **The whole prompt in one chunk** (`Geometry.prompt_rows`, a recipe's `prompt_rows` decl). Bidirectional attention
  across chunks cannot run chunk by chunk through the layers, so a PrefixLM prompt runs whole from position 0. The scratch
  rows follow the geometry (`model.scratchRows`). The prompt chunk ignores prompt-cache marks for a PrefixLM recipe,
  though this engine keeps no prompt states anyway. `pass.rowsOf` refuses a PrefixLM chunk that does not start at 0.
  `Core.open` refuses `--context` past the prompt rows by name (`PrefixPastPromptRows`).
- **Multi-head attention** (one KV head a query head). The dense engine built `pattn` at 8 or 2 query heads a block for
  head dim 128, so a group of 1 was refused. This adds `pattn_kernel<128, 8, 1, 8>` (one head a warp, 128 rows a block)
  and its PrefixLM twin, plus `pattn_prefix_kernel<128, 8, 2, 8>`. The one-head instance takes only a group of exactly 1
  (`exact`), so odd groups such as 3 still refuse as before.
- Host tests: the MHA and PrefixLM shapes and symbols, the refusals.

## 0002-hrm-text-family.patch

`zig/src/families/hrm_text/` (config, weights, kernels, recipe, a fixture of the release's `config.json`, and torch's
float32 `inv_freq`), `zig/kernels/cuda/hrm_text.cu`, and its wiring into the build and the CUDA registry (rebased onto
v1.3.7 alongside `lfm3`).

The forward is transformers 5.24's `HrmTextModel`, mapped onto the dense engine's layer loop:

- The engine's layer `i` (of `16 x 2 x (3 + 1) = 128`) is stack call `i / 16`'s layer `i % 16`. Calls 3 and 7 run the
  H module and the others run L. Each `i` keeps its own keys and values in slot `i`, the cache layout HF documents
  (`cycle_offset + layer_idx`).
- `z_H` starts as `embed(ids) * embedding_scale` (`tf_hrm_embed`), and the first input is `z_L_init + z_H`. In the
  release, `z_L_init` is a trained, nonzero vector, not zeros.
- At each call boundary, the previous call's last residual and its weightless final norm become the state it updates.
  The next input is always that new state plus the other one. Both states live in the weights' own bf16 row buffers, a
  prompt chunk wide.
- A layer: weightless RMS norm, the fused `attn.gqkv_proj` (`gate|q|k|v`), RoPE in torch's bf16 rounding order, paged
  attention over slot `i`, `sigmoid(gate) * attn`, `o_proj`, weightless RMS norm, `gate_up_proj` (`gate|up`), SwiGLU,
  `down_proj`.
- Weights are read as stored. A projection whose `weight` is `uint8` is ModelOpt NVFP4 (`weight_scale` E4M3 per 16
  inputs, `weight_scale_2` f32) and packs onto the lane matmul's FP4 mode with bf16 rows (W4A16; any `input_scale` is
  left unread). Anything else must be BF16. The embedding and `z_L_init` are BF16. `lm_head` is either.
- Admission (`config.zig`): `model_type hrm_text`, head dim 128, MHA, SiLU, no biases, an untied head, default rope, at
  most 16 calls of at most 32 layers, a window of at most 8192 for a PrefixLM, and a ModelOpt NVFP4 scheme or none.
- `reasoning = false` (the release's template renders thinking off). The `cuda_family_info.zig` test that only
  dots.mocr opts out of reasoning now names HRM-Text too.
- The dense engine is built once in `zig/build/cuda_engines.zig` and shared by dots.mocr and HRM-Text.

## 0003-cuda-dense-spark-perf.patch

DGX Spark (GB10, `sm_121`, 48 SMs) performance stack on `cuda_dense` and `hrm_text`:

- **Decode CUDA graph cache** (`cuda_dense/backend.zig`, `pass.zig`): splits `pass.forward` into host staging
  (`pass.upload`) and pure device launches (`pass.launch`), caching captured decode graphs via `cuda.graph_cache` for
  up to 16 concurrent streams keyed by `(n_chains, head0, heads, slots, lens)` and buffer fingerprint. Honors
  `TF_GRAPH_CACHE=0` to run eager. Bit-identical.
- **Blackwell PDL** (`cuda_dense/kernels.zig`, `hrm_text.cu`): sets `k.lane.pdl = true` on `sm_90+` (`TF_DENSE_PDL=1`
  by default) and emits `griddepcontrol.launch_dependents` at entry of every `hrm_text` glue kernel preceding a `qmmf`
  matmul (`tf_hrm_add_rms`, `tf_hrm_start_rms`, `tf_hrm_turn_rms`, `tf_hrm_silu_mul`, `tf_hrm_attn_merge_gate`,
  `tf_hrm_gate_strided`), so all 513 matmuls per decode step overlap their 3-stage weight loads with the preceding glue.
  Bit-identical.
- **Fast NVFP4 and BF16 prompt GEMMs** (`cuda_dense/kernels.zig`, `pass.zig`, `hrm_text/cuda_recipe.zig`):
  - Loads `cuda.nvfp4.Prompt` (`prompt16.cu` `prompt_lean` 128×256 persistent scale-folded FP4 tile + `lean_swiglu`
    fused gate/up SwiGLU) and dispatches NVFP4 prompt projections to it when `rows >= 64` and tiles fill more than half
    the SMs (`2 * tiles > sms`, keeping short prompts on the faster split-K `qmmf` lane and preserving 0.02–0.03 s TTFT).
    Gated by `TF_DENSE_PROMPT_FP4` (default `1`; `0` keeps the exact lane everywhere).
  - Calls `k.lane.loadPrompt(d)` (`TF_DENSE_PROMPT_BF16=1` by default) to enable sm_12x 128×128 BF16 K32/K64 prompt
    tiles (+39% to +48% BF16 prefill, bit-identical).
- **Fused `hrm_text` glue & MHA (`G = 1, HD = 128`) decode attention** (`hrm_text.cu`, `cuda_kernels.zig`,
  `cuda_recipe.zig`):
  - `tf_hrm_gqkv_rope_kv`: rotates `q`, rotates `k` and copies `v` directly into paged `KC`/`VC` (eliminating 128
    `tf_nemo_kv_write_segs` launches and the `x.kv` round-trip), and skips copying `gate` (read in place from `x.wide` at
    stride `4 * D`).
  - `tf_hrm_attn_chunk_mha` + `tf_hrm_attn_merge_gate`: replaces generic `GMAX=16` `tf_nemo_attn_chunk_segs` /
    `tf_nemo_attn_merge_segs` + `tf_hrm_gate` with a bit-identical 128-thread `G = 1` MHA kernel using 16-byte coalesced
    K/V loads, 1 thread/dim PV, and a fused sigmoid output gate epilogue (eliminating 128 `tf_hrm_gate` launches and all
    unused `XS` group sums).
  - `tf_hrm_add_rms`, `tf_hrm_start_rms`, `tf_hrm_turn_rms`, and vectorized `tf_hrm_silu_mul`: weightless RMS norm
    (no `ones` reads), 3-into-1 stack-boundary fusion (14 fewer launches/token), and 16-byte vectorized SwiGLU, all
    bit-identical and PDL-triggering.
- **Note on `TF_QMMF_HALF_BN`**: tested via live ablation on GB10 and intentionally left off (`0`). Halving `o_proj`
  `BN` from 64 to 32 doubled activation reads and regressed 4-stream decode by ~3% (193.4 → 187.0 tok/s on Danish).

## Not done

- No launch records in `zig/src/core/contract/` for the new kernels, so the contract runner (`tf-cuda-test`) does not
  cover them.
- No structured output: the dense engine refuses grammar-masked windows (`StructuresUnsupported`), so `response_format`
  and `tool_choice: "required"` are refused. Plain tool calls are parsed from the reply text by the server.
- No prompt cache across turns. With PrefixLM, an earlier turn's keys depend on the later prompt tokens, so they
  cannot be reused.

Host tests (no GPU needed):

```bash
zig build test-cuda-dense-host test-hrm-text-cuda-host test-dots-mocr-cuda-host check-native-cuda-host
```
