<h1 align="center">DFM-Mimir v1.8 (2B) NVFP4 on one DGX Spark (TensorFold)</h1>

Serve **DFM-Mimir v1.8** from Danish Foundation Models on a single NVIDIA DGX Spark (GB10) through an OpenAI-compatible
API. The checkpoint is [`danish-foundation-models/DFM-Mimir-v1.8`](https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.8)
at revision `30485dba`: the XL EMA checkpoint at step 3.8M, 1.79B parameters, BF16, Apache-2.0. The recipe serves it
with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (release
[v1.3.4](https://github.com/ashhart/TensorFold/releases/tag/v1.3.4), commit `12ebd414`), plus the two patches in
[`patches/`](patches/README.md), inside NVIDIA's PyTorch container, on one GPU.

```bash
./start.sh       # build the image, download (~3.6 GB), export NVFP4, serve on port 8890, smoke test
./stop.sh
```

> **Status: built, verified and benchmarked** on one DGX Spark (2026-11-25). The NVFP4 engine matches transformers
> running the export's own decoded weights token for token on every check prompt. The BF16 engine matches the BF16
> release, apart from greedy ties (see [Checking it](#checking-it)). NVFP4 decodes at about 49 tok/s on one stream and
> 175 tok/s across four, twice BF16's speed (see [Performance](#performance)).

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
attention, lanes, sampling and the server, so the patches add what it lacks and the family on top. In short:

- **0001**: bidirectional prompt attention, the whole prompt in one chunk (up to 8,192 rows), and a one-head-a-warp
  attention instance for multi-head attention.
- **0002**: the `hrm_text` recipe (the recurrence over 128 cache slots, gate|q|k|v RoPE, the output gate, scaled
  embedding) and its BF16 and ModelOpt NVFP4 weight loading.

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

These checks were run on 2026-11-25 with the results given under each step. Rerun them after changing a patch or the
exporter:

1. **Build and the host tests.** `scripts/prepare.sh --image-only` builds the patched engine. For the patches' unit tests
   (no GPU), run `zig build test-cuda-dense-host test-hrm-text-cuda-host test-dots-mocr-cuda-host` in a patched checkout.
   Result: 73 of 73 pass, and the engine builds with its kernels.
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
   Result: 3 of 5 replies are identical. The other two (Danish at token 103 and English at token 11) leave the reference
   where transformers' own top two logits are exactly equal in BF16 (15.875 and 15.875; 15.250 and 15.250). For the
   English one, transformers' own teacher-forced forward picks the engine's token, not the one its `generate` emitted.
   These are ties, not errors.
3. **NVFP4**: `./start.sh restart`, then `python3 tools/check.py --reference work/reference.json` again. NVFP4 replies
   are reported against the BF16 reference rather than judged. To check the FP4 path itself, run
   `python3 tools/nvfp4_reference.py --model <release snapshot> --export <NVFP4 export folder>` (the same Python as
   step 2). It runs transformers with each projection replaced by the export's decoded NVFP4 weights (E2M1 code ×
   E4M3 scale × global) and compares greedy replies with the server. Result: all 5 prompts are identical to the
   engine, token for token (128, 56, 28, 3 and 8 tokens). Against the BF16 release, the same 3 of 5 agree in full, and
   the English reply leaves it where the BF16 one did. The Danish reply leaves it at character 65 by putting
   "**kædning** (chaining)" where BF16 puts "\"chaining\" (kædning)". Weight-only RTN on a model that reuses each
   layer 2 to 6 times a token is the main quality risk of this recipe, so compare on your own tasks too.

`tools/check.py` without `--reference` checks that the engine agrees with itself:

- drafted replies equal undrafted ones
- concurrent replies equal solo ones
- a 4,710-token prompt finds a code word hidden in its middle (this needs the one-chunk PrefixLM path)
- a tool call comes back as `tool_calls`

## Performance

Measured on one DGX Spark (GB10, 121 GiB visible) on 2026-11-25 with `python3 tools/bench.py`:

- Settings: the defaults (`--context 8192 --parallel 4`, context-copy drafts on). 256 tokens a reply, thinking off,
  `ignore_eos`.
- Each cell is the median of 3 runs.
- Decode speed is the server's own `tokens_per_second`, with the first token excluded.
- Aggregate is the sum over the concurrent streams.

Decode, greedy:

| Prompt | NVFP4, 1 stream | NVFP4, 4 streams (aggregate) | BF16, 1 stream | BF16, 4 streams (aggregate) | Accepted tokens a round (NVFP4) |
| --- | --- | --- | --- | --- | --- |
| Danish prose | 49.1 tok/s | 173.6 tok/s | 23.3 tok/s | 87.2 tok/s | 1.00 |
| English prose | 49.5 tok/s | 174.9 tok/s | 28.6 tok/s | 105.2 tok/s | 1.00 (1.24 as BF16) |
| Python code | 63.8 tok/s | 215.3 tok/s | 23.6 tok/s | 88.5 tok/s | 1.30 |

- **NVFP4 is about twice as fast as BF16.** Each token runs 128 layer applications, so it reads every stack's weights
  6 or 2 times. That is about 8 GB a token as BF16, and the decode is bound by memory bandwidth.
- **Drafts.** The drafts copy text from the prompt, so they only pay when the reply repeats it. Code accepts 1.30
  tokens a round, worth 1.3x; prose accepts about none. Without drafts (`DRAFTS=0`), every prompt decodes at 48.9 to
  49.4 tok/s on one stream and 173 to 176 tok/s across four, so drafts cost nothing where they do not help.
- **Sampling.** At `T=0.8` (seeded), decode runs at 48.5 to 49.8 tok/s on one stream and 169 to 175 tok/s across
  four. Code's drafts mostly stop being accepted.
- **More streams** (`./start.sh restart --parallel 16`): 8 streams reach 287 to 342 tok/s aggregate (36 to 43 tok/s
  each), and 16 streams reach 409 to 480 tok/s (26 to 30 tok/s each). At 16 streams with full 8K windows, the KV cache
  can grow to 102 GiB, which is why the default stays at 4.

Prefill, cold (fresh random Danish and English text every run). PrefixLM runs a prompt whole, in one chunk, with
nothing cached:

| Prompt | NVFP4 | BF16 |
| --- | --- | --- |
| 1k tokens | 0.23 s, 4,567 tok/s | 0.29 s, 3,591 tok/s |
| 2k tokens | 0.49 s, 4,243 tok/s | 0.60 s, 3,450 tok/s |
| 4k tokens | 1.10 s, 3,760 tok/s | 1.32 s, 3,119 tok/s |
| 8k tokens | 2.58 s, 3,135 tok/s | 2.98 s, 2,690 tok/s |

Time to first token on the short decode prompts is 0.02 to 0.03 s for one stream and about 0.07 s for four. Because
nothing is cached, every turn of a conversation prefills its whole history again, which takes about 2.6 s near the
8K limit. The server loads in about 2 s.

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
| `TENSORFOLD_API_KEY` | unset | require a bearer token |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 | MemAvailable the engine leaves free |

Extra `tensorfold-native serve` arguments go after the defaults and win: `./start.sh restart --parallel 8`.
`DRY_RUN=1 ./start.sh` prints the docker command. `scripts/prepare.sh --reexport` makes the export again.

## Disk

The image (`tensorfold-mimir18:v1.3.4-12ebd41`) adds about 1 GB to NVIDIA's `pytorch:26.07-py3` base, which the other
TensorFold recipes on this Spark already pulled. The release takes 3.6 GB under `HF_CACHE`. The NVFP4 export takes
2.1 GB (3.4 GB for `FORMAT=bf16`).

## Credits and licenses

- DFM-Mimir v1.8: Danish Foundation Models, weights under Apache-2.0. Tokenizer origins (Gemma) carry their own terms.
  The architecture is [HRM-Text](https://github.com/schneiderkamplab/HRM-Text).
- TensorFold: by ashhart, under its own license, which is copied into the image at `/opt/tensorfold/share/doc/tensorfold`.
- libtfgrammar: from TensorFold's release archive, with xgrammar's Apache-2.0 notices.
- The patches in `patches/` modify TensorFold source and carry its license. `tools/reference.py` runs transformers'
  `HrmTextForCausalLM` (Apache-2.0).
