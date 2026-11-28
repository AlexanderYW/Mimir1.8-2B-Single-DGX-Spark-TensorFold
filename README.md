<h1 align="center">DFM-Mimir v1.8 (2B) NVFP4 on one DGX Spark (TensorFold)</h1>

Serve **DFM-Mimir v1.8** from Danish Foundation Models on a single NVIDIA DGX Spark (GB10) through an OpenAI-compatible
API. The checkpoint is [`danish-foundation-models/DFM-Mimir-v1.8`](https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.8)
at revision `30485dba`: the XL EMA checkpoint at step 3.8M, 1.79B parameters, BF16, Apache-2.0. The recipe serves it
with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (release
[v1.3.7](https://github.com/ashhart/TensorFold/releases/tag/v1.3.7), commit `b3366549`), plus the three patches in
[`patches/`](patches/README.md), inside NVIDIA's PyTorch container, on one GPU.

```bash
./start.sh       # build the image, download (~3.6 GB), export NVFP4, serve on port 8890, smoke test
./stop.sh
```

> **Status: built, verified and benchmarked** on one DGX Spark (2026-11-28 on v1.3.7). Both the NVFP4 and BF16 builds
> pass all self-consistency checks (drafted == serial, concurrent == alone, 4,710-token needle, tool calling) and match
> the transformers BF16 reference identically to v1.3.4/v1.3.5 (see [Checking it](#checking-it)). NVFP4 decodes at
> **56 tok/s** on one prose stream (**72.5 tok/s** on code) and **184–234 tok/s** across four streams, with cold
> prefill reaching **6,134 tok/s** at 2k and **4,397 tok/s** (1.83 s) at 8k (see [Performance](#performance)).

## The model, and why the patches

Mimir v1.8 is an **HRM-Text** model (`model_type hrm_text`, transformers 5.24's `HrmTextForCausalLM`), not a plain
decoder:

- **Two stacks of 16 layers**, L and H, run recurrently: `z_H = embed * 39.19`, `z_L = z_L_init`; twice, three times
  `z_L = L(z_L + z_H)` then `z_H = H(z_H + z_L)`. That is 8 stack calls, **128 layer applications a token**, each with its
  own keys and values. Compute a token is that of a ~4.3B dense model, from 1.79B parameters.
- Layers are Llama-shaped with **weightless RMS norms**, **multi-head attention** (12 heads of 128) with a **sigmoid
  output gate**, and SwiGLU 1536 → 4096. The checkpoint fuses `gate|q|k|v` and `gate|up`.
- **PrefixLM**: the whole prompt attends bidirectionally, and the reply causally. Prompts cannot be chunked or cached
  across turns.
- Gemma 4's 262,144-token vocabulary and native chat template (tool calls via `<|tool_call>`, thinking off).

Stock TensorFold has no `hrm_text` family. Its shared CUDA dense engine (the one dots.mocr runs on) already does paged
attention, lanes, sampling and the server, so the patches add what it lacks, the family, and Spark-specific performance
optimizations:

- **0001**: bidirectional prompt attention, the whole prompt in one chunk (up to 8,192 rows), and a one-head-a-warp
  attention instance for multi-head attention.
- **0002**: the `hrm_text` recipe (the recurrence over 128 cache slots, gate|q|k|v RoPE, the output gate, scaled
  embedding) and its BF16 and ModelOpt NVFP4 weight loading.
- **0003**: DGX Spark (GB10) performance stack for `cuda_dense` and `hrm_text` — CUDA graph capture/replay for decode
  rounds (`cuda.graph_cache`), Blackwell PDL (`TF_DENSE_PDL=1`) with `griddepcontrol.launch_dependents` across all
  preceding glue kernels, SM-gated persistent 128×256 scale-folded NVFP4 prompt GEMMs + fused `gate_up` SwiGLU
  (`nvfp4.Prompt`), sm_12x 128×128 BF16 K32/K64 prompt tiles (`Lane.loadPrompt`), fused RoPE + direct paged KV writes,
  fused weightless RMS/stack-turn norms and vectorized SwiGLU, and a dedicated 128-thread MHA (`G = 1, HD = 128`)
  decode attention kernel with a fused sigmoid output gate epilogue.

[`patches/README.md`](patches/README.md) gives the details and what is not done.

## Why an export step

No NVFP4 release of v1.8 exists on the Hub (only BF16 and GGUF). `scripts/prepare.sh` therefore makes one with
[`scripts/nvfp4_export.py`](scripts/nvfp4_export.py), inside the recipe's image with no network and on the CPU by
default:

| Tensor | Stored as | |
| --- | --- | --- |
| 128 stack projections (0.98B parameters) | NVFP4, ModelOpt layout: E2M1 codes, an E4M3 scale per 16 inputs, an f32 global | weights only, round-to-nearest, no calibration; the engine multiplies them by bf16 rows (W4A16) |
| `embed_tokens`, `z_L_init` | BF16 | as released |
| `lm_head` (0.40B) | BF16 by default; `HEAD_FORMAT=nvfp4` packs it too | ModelOpt's own default leaves the head unquantized |

The export also writes `max_position_embeddings` (`WINDOW`, default 8192). The model card's own evaluation serves 8192
with plain RoPE through vLLM's `--hf-overrides`; the release names 4096, which the export records as
`hrm_trained_max_position_embeddings`. Use `WINDOW=4096` to stay inside the trained window. The export adds a
`generation_config.json` that stops on `<turn|>`, `<|tool_response>` (a tool-calling turn hands over there) and
`<eos>`. Files go to `~/.cache/huggingface/tensorfold-models/DFM-Mimir-v1.8-nvfp4-head-bf16-w8192-30485dba-<hash>`.
The hash covers the settings and the exporter, so a change makes a new folder. `FORMAT=bf16` exports the weights as
stored instead.

In memory, the engine reports 2.78 GiB of model for the NVFP4 export and 4.09 GiB for BF16. The NVFP4 stacks take
0.54 GiB, against 1.83 GiB as BF16, and the embedding and head take 0.75 GiB each. The rest is the two states and an
8,192-row prompt's scratch.

## What it serves

| | |
| --- | --- |
| Model | `mimir-1.8` (`SERVED_NAME`), the NVFP4 export of `danish-foundation-models/DFM-Mimir-v1.8@30485dba` |
| Window | 8,192 tokens (`CONTEXT`, at most `WINDOW`): prompt and reply together. A prompt runs whole in one chunk |
| KV cache | 768 KiB a token (128 slots × 12 heads × 128 × K,V × bf16): a full 8K window is 6.4 GiB, in 384 MiB pages |
| Streams | up to 4 at once (`PARALLEL`, at most 128); pages are leased as streams grow |
| Drafts | context copies from the prompt's own text (no draft head); verified, so replies are the same with `DRAFTS=0` |
| Chat | the release's template, thinking off (`--no-thinking`); tool calls parsed from Gemma 4's markers |
| Sampling | greedy by default (`TEMPERATURE=0`, as the release's evaluations decode); a request's own values win |

Not served: structured output (`response_format`, `tool_choice: "required"`), because the dense engine has no grammar
masks yet. Also not served: prompt caching across turns, which PrefixLM rules out; tensor parallel; images.

## Checking it

These checks were run on 2026-11-28 on TensorFold v1.3.7 with patches `0001`–`0003` (both NVFP4 and BF16 exports), with
the results given under each step. Rerun them after changing a patch or the exporter:

1. **Build and the host tests.** `scripts/prepare.sh --image-only` builds the patched engine. For the patches' unit tests
   (no GPU), run `zig build test-cuda-dense-host test-hrm-text-cuda-host test-dots-mocr-cuda-host` and
   `zig build check-native-cuda-host` in a patched checkout.
   Result: all host tests pass, every touched Zig file passes `zig ast-check` and `zig fmt --check`, and the engine
   builds with its kernels.
2. **The forward against transformers**, on the BF16 export, where the replies should match token for token, ties
   aside:
   ```bash
   # any Python with torch on the GPU and transformers 5.24+ (which has HrmTextForCausalLM)
   python3 tools/reference.py \
     --model ~/.cache/huggingface/hub/models--danish-foundation-models--DFM-Mimir-v1.8/snapshots/30485dbad6f3ce50d09eaa963b8a8fce53a4290c \
     --out work/reference.json
   FORMAT=bf16 ./start.sh
   python3 tools/check.py --reference work/reference.json
   ```
   Result (v1.3.7): 3 of 5 replies are identical (`fakta`, `regning`, `lang`), and the other two (`dansk` at char 362 /
   token 103, `english` at char 66 / token 11) leave the reference at the exact same BF16 greedy ties as v1.3.4 (15.875
   vs 15.875; 15.250 vs 15.250), confirming bit-exactness of the fused BF16 + graph + PDL path.
3. **NVFP4**: `./start.sh restart`, then `python3 tools/check.py --reference work/reference.json` again. NVFP4 replies
   are reported against the BF16 reference rather than judged.
   Result (v1.3.7): all self-consistency checks pass (`drafted == serial`, `concurrent == alone`, 4,710-token needle
   `'BLÅBÆR-47'`, tool call). Against the BF16 reference, the exact same 3 of 5 replies agree in full, `english` leaves
   at character 66, and `dansk` leaves at character 65 (`"**kædning** (chaining)"` vs `"\"chaining\" (kædning)"`),
   identical to v1.3.4/v1.3.5. On short prompts (< 64 rows), NVFP4 stays on the exact `qmmf` lane; on longer prompts
   filling > half the SMs, `TF_DENSE_PROMPT_FP4=1` uses TensorFold's persistent scale-folded `prompt_lean` / `lean_swiglu`
   tiles (`TF_DENSE_PROMPT_FP4=0` keeps the exact lane everywhere). To check the exact FP4 weight decoding path against
   transformers, run `python3 tools/nvfp4_reference.py --model <release snapshot> --export <NVFP4 export folder>`.
   Weight-only RTN on a model that reuses each layer 2 to 6 times a token is the main quality risk of this recipe, so
   compare on your own tasks too.

`tools/check.py` without `--reference` checks that the engine agrees with itself:

- drafted replies equal undrafted ones
- concurrent replies equal solo ones
- a 4,710-token prompt finds a code word hidden in its middle (this needs the one-chunk PrefixLM path)
- a tool call comes back as `tool_calls`

## Performance

Measured on one DGX Spark (GB10, 121 GiB visible) on 2026-11-28 on **TensorFold v1.3.7 + patches 0001–0003** with
`python3 tools/bench.py`:

- Settings: the defaults (`--context 8192 --parallel 4`, context-copy drafts on, CUDA graphs + PDL + prompt tiles on).
  256 tokens a reply, thinking off, `ignore_eos`.
- Each cell is the median of 3 runs.
- Decode speed is the server's own `tokens_per_second`, with the first token excluded.
- Aggregate is the sum over the concurrent streams.

Decode, greedy:

| Prompt | NVFP4, 1 stream | NVFP4, 4 streams (aggregate) | BF16, 1 stream | BF16, 4 streams (aggregate) | Accepted tokens a round (NVFP4) |
| --- | --- | --- | --- | --- | --- |
| Danish prose | **55.8 tok/s** *(was 49.9)* | **183.8 tok/s** *(was 175.8)* | **25.9 tok/s** *(was 23.3)* | **94.9 tok/s** *(was 87.2)* | 1.00 |
| English prose | **55.9 tok/s** *(was 50.1)* | **186.3 tok/s** *(was 177.0)* | **31.7 tok/s** *(was 28.6)* | **115.3 tok/s** *(was 105.2)* | 1.00 (1.24 as BF16) |
| Python code | **72.5 tok/s** *(was 64.4)* | **234.0 tok/s** *(was 217.8)* | **26.3 tok/s** *(was 23.6)* | **96.4 tok/s** *(was 88.5)* | 1.30 |

*(Baseline comparisons in italics are v1.3.5 for NVFP4 and v1.3.4 for BF16.)*

- **+11% to +13.6% faster decode** across both NVFP4 and BF16, 100% bit-identical: fusing RoPE + paged KV writes,
  specialized 128-thread MHA (`G = 1`) decode attention + fused sigmoid output gate, weightless RMS + stack-turn norm
  fusion, Blackwell PDL across all 513 matmuls/token, and CUDA graph capture/replay cut ~300 kernel launches and CPU
  launch overhead per step.
- **NVFP4 is more than twice as fast as BF16.** Each token runs 128 layer applications, reading every stack's weights
  6 or 2 times (~8 GB a token as BF16 vs ~3.0 GB as NVFP4 including BF16 `lm_head`).
- **Drafts.** The drafts copy text from the prompt, so they only pay when the reply repeats it. Code accepts 1.30
  tokens a round, worth 1.3×; prose accepts about none.
- **More streams** (`./start.sh restart --parallel 16`): at 16 streams with full 8K windows, the KV cache can grow to
  102 GiB, which is why the default stays at 4. Decode CUDA graphs cover up to 16 concurrent streams.

Prefill, cold (fresh random Danish and English text every run). PrefixLM runs a prompt whole, in one chunk, with
nothing cached:

| Prompt | NVFP4 | BF16 |
| --- | --- | --- |
| 1k tokens | **0.17 s, 6,063 tok/s** *(was 0.23 s, 4,608)* | **0.20 s, 5,306 tok/s** *(was 0.29 s, 3,591)* |
| 2k tokens | **0.34 s, 6,134 tok/s** *(was 0.49 s, 4,264)* | **0.41 s, 5,122 tok/s** *(was 0.60 s, 3,450)* |
| 4k tokens | **0.76 s, 5,424 tok/s** *(was 1.09 s, 3,791)* | **0.91 s, 4,541 tok/s** *(was 1.32 s, 3,119)* |
| 8k tokens | **1.83 s, 4,397 tok/s** *(was 2.55 s, 3,157)* | **2.14 s, 3,747 tok/s** *(was 2.98 s, 2,690)* |

- **+32% to +45% faster NVFP4 prefill** and **+39% to +48% faster BF16 prefill**: persistent 128×256 `prompt_lean` +
  fused `lean_swiglu` (NVFP4), sm_12x 128×128 K32/K64 prompt tiles (BF16, bit-exact), and eliminating the intermediate
  `kv` and `gate` copies (~19 GB less DRAM traffic on an 8k prompt) cut an 8k NVFP4 prompt from 2.55 s down to **1.83 s**.
- Time to first token on the short decode prompts stays at **0.02 to 0.03 s** for one stream and **0.05 to 0.07 s** for
  four (SM-fill gating keeps short prompts on the exact split-K `qmmf` lane). The server loads in 2 to 4 s.

## Settings

Everything lives in [`scripts/config.sh`](scripts/config.sh). The environment wins over `./.env`; see
[`.env.example`](.env.example).

| Setting | Default | |
| --- | --- | --- |
| `PORT`, `HOST`, `SERVED_NAME` | 8890, 0.0.0.0, mimir-1.8 | |
| `FORMAT`, `HEAD_FORMAT` | nvfp4, bf16 | what the export stores |
| `WINDOW` | 8192 | the export's `max_position_embeddings`, at most 8192 |
| `CONTEXT` | `WINDOW` | at most `WINDOW` |
| `PARALLEL` | 4 | 1 to 128; a cap |
| `MAX_TOKENS` | 2048 | |
| `THINKING` | 0 | the template's `enable_thinking` |
| `DRAFTS` | 1 | 0 passes `--no-drafts` |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | 0, 1, 0 | |
| `EXPORT_DEVICE` | cpu | `cuda` runs the export on the GPU |
| `TF_GRAPH_CACHE` | 1 | 0 disables dense decode CUDA graphs (runs eager) |
| `TF_DENSE_PDL` | 1 | 0 disables Blackwell PDL on `qmmf` decode lanes |
| `TF_DENSE_PROMPT_FP4` | 1 | 0 keeps exact `qmmf` lane on NVFP4 prompts |
| `TF_DENSE_PROMPT_BF16` | 1 | 0 disables sm_12x 128×128 BF16 prompt tiles |
| `TENSORFOLD_API_KEY` | unset | require a bearer token |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 | MemAvailable the engine leaves free |

Extra `tensorfold-native serve` arguments go after the defaults and win: `./start.sh restart --parallel 8`.
`DRY_RUN=1 ./start.sh` prints the docker command. `scripts/prepare.sh --reexport` makes the export again.

## Disk

The image (`tensorfold-mimir18:v1.3.7-b336654`) adds about 1 GB to NVIDIA's `pytorch:26.07-py3` base, which the other
TensorFold recipes on this Spark already pulled. The release takes 3.6 GB under `HF_CACHE`. The NVFP4 export takes
2.1 GB (3.4 GB for `FORMAT=bf16`).

## Credits and licenses

- DFM-Mimir v1.8: Danish Foundation Models, weights under Apache-2.0. Tokenizer origins (Gemma) carry their own terms.
  The architecture is [HRM-Text](https://github.com/schneiderkamplab/HRM-Text).
- TensorFold: by ashhart, under its own license, which is copied into the image at `/opt/tensorfold/share/doc/tensorfold`.
- libtfgrammar: from TensorFold's release archive, with xgrammar's Apache-2.0 notices.
- The patches in `patches/` modify TensorFold source and carry its license. `tools/reference.py` runs transformers'
  `HrmTextForCausalLM` (Apache-2.0).
