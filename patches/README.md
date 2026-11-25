# Patches

Applied in order with `git apply` on TensorFold [v1.3.4](https://github.com/ashhart/TensorFold/releases/tag/v1.3.4)
(`12ebd414`) when `scripts/prepare.sh` builds the image. Both apply cleanly to a fresh v1.3.4 checkout
(`git apply --check`), and every touched Zig file passes `zig ast-check` and `zig fmt --check` with Zig 0.17.0. The
patched tree builds (`fatbins native`, `-Doptimize=fast`), its host tests pass (73 of 73), and the served forward
matches transformers (the main README's "Checking it").

Stock v1.3.4 has no `hrm_text` family, so `tensorfold-native serve` refuses Mimir v1.8 with "the native CUDA engine has
no backend for hrm_text checkpoints yet". HRM-Text is close to a dense decoder, which TensorFold's shared CUDA dense
engine (`zig/src/cuda_dense`, the one dots.mocr runs on) already serves: paged KV, lanes, draws, context-copy drafts and
the OpenAI server. The patches add the three things it lacks, then the family as a dense recipe.

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
float32 `inv_freq`), `zig/kernels/cuda/hrm_text.cu`, and its wiring into the build and the CUDA registry.

The forward is transformers 5.24's `HrmTextModel`, mapped onto the dense engine's layer loop:

- The engine's layer `i` (of `16 x 2 x (3 + 1) = 128`) is stack call `i / 16`'s layer `i % 16`. Calls 3 and 7 run the
  H module and the others run L. Each `i` keeps its own keys and values in slot `i`, the cache layout HF documents
  (`cycle_offset + layer_idx`).
- `z_H` starts as `embed(ids) * embedding_scale` (`tf_hrm_embed`), and the first input is `z_L_init + z_H`
  (`tf_hrm_start`). In the release, `z_L_init` is a trained, nonzero vector, not zeros.
- At each call boundary, the previous call's last residual and its weightless final norm become the state it updates.
  The next input is always that new state plus the other one (`tf_hrm_turn`). Both states live in the weights' own
  bf16 row buffers, a prompt chunk wide.
- A layer: weightless RMS norm (the engine's norm times a bf16 ones vector, which is exact); the fused
  `attn.gqkv_proj` (rows are gate | q | k | v, as transformers' conversion mapping chunks them); RoPE in torch's bf16
  rounding order (`tf_hrm_gqkv_rope`); paged attention over slot `i`; `sigmoid(gate) * attn` (`tf_hrm_gate`); `o_proj`;
  weightless RMS norm; `gate_up_proj` (gate | up); SwiGLU; `down_proj`.
- Weights are read as stored. A projection whose `weight` is `uint8` is ModelOpt NVFP4 (`weight_scale` E4M3 per 16
  inputs, `weight_scale_2` f32) and packs onto the lane matmul's FP4 mode with bf16 rows (W4A16; any `input_scale` is
  left unread). Anything else must be BF16. The embedding and `z_L_init` are BF16. `lm_head` is either.
- Admission (`config.zig`): `model_type hrm_text`, head dim 128, MHA, SiLU, no biases, an untied head, default rope, at
  most 16 calls of at most 32 layers, a window of at most 8192 for a PrefixLM, and a ModelOpt NVFP4 scheme or none.
- `reasoning = false` (the release's template renders thinking off). The `cuda_family_info.zig` test that only
  dots.mocr opts out of reasoning now names HRM-Text too.
- The dense engine is now built once in `zig/build/cuda_engines.zig` and shared by dots.mocr and HRM-Text. Zig refuses
  one source file in two modules of a single compilation, so each recipe could not make its own.

## Not done

- No launch records in `zig/src/core/contract/` for the new kernels, so the contract runner (`tf-cuda-test`) does not
  cover them.
- No structured output: the dense engine refuses grammar-masked windows (`StructuresUnsupported`), so `response_format`
  and `tool_choice: "required"` are refused. Plain tool calls are parsed from the reply text by the server.
- No prompt cache across turns. With PrefixLM, an earlier turn's keys depend on the later prompt tokens, so they
  cannot be reused.

Host tests (no GPU needed):

```bash
zig build test-cuda-dense-host test-hrm-text-cuda-host test-dots-mocr-cuda-host
```
