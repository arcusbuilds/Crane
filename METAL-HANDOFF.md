# Qwen3.5 Metal performance investigation

## Start here: prompt for Codex on the Mac

> Read this entire handoff, the repository instructions, and the referenced source before changing code. Help me investigate Qwen3.5-0.8B performance on Apple Silicon. First inspect my Mac, tools and available model, build a Metal release binary, and measure an unchanged baseline. Capture an Instruments trace and use it to rank the candidates below. Then propose the smallest supported optimization and explain the evidence. Teach me one step at a time and involve me before changing scope. Preserve low-precision numerical behavior and masked-attention fallbacks. Keep measurements and conclusions in local artifacts. Do not claim a speedup without paired measurements. Do not push, open a PR, or post public comments without my explicit approval of the exact changes or draft.

## Scope and branch

- Related issue: https://github.com/lucasjinreal/Crane/issues/69 . The original report concerns Mac performance.
- Handoff branch: `docs/qwen35-metal-handoff`.
- Pinned source baseline: `c9640623cda94709606188f48176c31b8355260f`.
- This branch adds documentation only. It deliberately excludes CUDA PR https://github.com/lucasjinreal/Crane/pull/194 . Metal implementation work should use a new branch from this handoff, in the same checkout.
- These are source-backed hypotheses. No Mac baseline, trace, or speedup has been measured for this investigation. Verify the paths and dependency version again if rebasing to newer upstream.
- Start with dense safetensors Qwen3.5-0.8B, one request at a time. Treat GGUF, quantization and concurrency as separate workloads.

Once this branch has been pushed to the fork, a fresh Mac checkout can use:

```sh
git clone --branch docs/qwen35-metal-handoff --single-branch https://github.com/arcusbuilds/Crane.git
cd Crane
```

If Crane is already present, use that checkout and fetch the handoff branch from the fork; do not create a duplicate checkout. Check for local changes before switching. Keep the handoff branch intact and create an implementation branch when ready:

```sh
git switch -c perf/qwen35-metal-investigation
```

The Windows user requires public Git identity `212402043+arcusbuilds@users.noreply.github.com`, no assistant attribution in commits, and explicit approval before any push. Show the exact diff and verification results first. Draft public comments for review and post only after an explicit "post" for that draft. Keep the CUDA PR unchanged; any Metal PR will be separate and also needs approval.

## What we learned on CUDA

The separate CUDA PR fused single-token GDN convolution and reused existing preparation/post-normalization kernels. Five paired rounds, 100 measured requests, Qwen3.5-0.8B BF16 on an RTX 2000 Ada Laptop GPU measured median end-to-end output throughput 62.4 to 89.4 tokens/s. All tested output text matched. Baseline round medians varied about 11%; paired gains were 36-50%.

A short-context Nsight capture showed 1,235 to 659 GPU launches/token. These Windows results do not predict Mac performance. Metal already uses fused GDN stages; repeating the CUDA convolution change is not the starting point.

## Candidate 1: native Metal attention

Start at `crane-core/src/models/qwen3_5/modeling.rs`:

- `FullAttention::forward` around line 595 selects the attention route around 697-703.
- `grouped_sdpa_slice` around 780 converts low-precision Q, K and V to F32 around 794-796, including the entire occupied K/V cache each token.
- It then runs separate QK matmul, softmax and PV matmul, and converts the output back.

Candle 0.11 already supplies `candle_nn::ops::sdpa`. Inspect the resolved dependency's `src/ops.rs`, especially `Sdpa::metal_fwd` around 1033 and public `sdpa` around 1308. Its vector path supports the relevant head dimension, GQA and low-precision types, with different implementations below/above 1,024 cached positions. Confirm actual tensor layouts and dispatch eligibility before using it.

First experiment: supported Metal, single-query, **no explicit attention mask**, using the existing native SDPA path and preserving the generic fallback. The dependency's vector branch does not apply an arbitrary supplied mask; blindly replacing all calls would be incorrect. Confirm scaling, output shape, dtype, strides and device behavior. Do not expand this first experiment to masked prefill or every architecture.

Expected evidence: fewer cache casts and attention dispatches, particularly as occupied context grows. Potential numerical differences from accumulation order require explicit validation; neither exact logits nor faster execution is assumed.

## Candidate 2: ordered SwiGLU

`Mlp::forward` in `modeling.rs` around 917 calls `crane-core/src/ops/float_math.rs::swiglu_low_precision` around 281. On Metal the low-precision graph computes:

1. Convert gate to F32 and compute SiLU.
2. Round SiLU to the input dtype.
3. Multiply by up, then round the product to the output dtype.

A fused kernel must preserve both rounding points. The existing generic Metal SwiGLU is not a direct numerical substitute: it rounds at a different point. Inspect `crane-core/src/ops/fused_ops/swiglu.rs` and `crane-core/kernels/metal/fused_ops.metal` before adding plumbing.

For 24 MLP layers, reducing four materializing operations to one could remove up to 72 dispatches/token. This is an estimated dispatch reduction, not a measured speed gain. Test F16 and supported BF16, boundary values, offsets, layout handling and the existing graph as reference.

## Candidate 3: RMSNorm cast fusion

`Qwen35RmsNorm::forward` in `modeling.rs` around 80 calls `float_math.rs::rms_norm_low_precision` around 198. The Metal route casts input to F32, uses F32 RMSNorm and scale, then casts back. The normalization itself is already fused.

A mixed-dtype kernel could remove the two surrounding casts. Qwen3.5-0.8B has 61 such calls/token, suggesting up to 122 fewer dispatches. Preserve F32 accumulation and scale; casting the scale to half changes the computation. Reuse existing Metal launcher utilities and buffer pooling. Test F16/BF16 where supported, actual hidden/head widths, epsilon, tiny values and rounding-sensitive scales.

Partial RoPE is a lower-priority follow-up in `apply_mrope` around 339. It also has intermediate rounding requirements. Use trace evidence before expanding scope.

## First session on the Mac

Check the environment first; do not install or download anything silently:

```sh
sw_vers
uname -m
system_profiler SPHardwareDataType SPDisplaysDataType
rustc --version
cargo --version
xcode-select -p
xcrun xctrace list templates
git rev-parse HEAD
```

Record chip, RAM, macOS/Xcode/Rust versions and resolved Candle version. Full Xcode supplies Instruments; Command Line Tools alone may not supply the needed templates. Locate an existing checkpoint first; if absent, agree on download location and size. Record the exact model revision and weight format.

The repository documents this release build:

```sh
cargo build -p crane-serve --release --features "metal,accelerate"
```

Preserve a copy of the baseline binary outside `target` before later builds overwrite it. Start with F16, the documented Qwen3.5 Metal default; BF16 is a separate run if supported by the actual hardware/software. A dtype change is not an optimization comparison.

```sh
MODEL_PATH="/absolute/path/to/Qwen3.5-0.8B"
CRANE_PROF=0 ./target/release/crane-serve \
  --model-path "$MODEL_PATH" --model-type qwen3_5 --dtype f16 \
  --host 127.0.0.1 --port 18069 --max-concurrent 1 --context 2048
```

Confirm logs say Metal and the intended dtype. Once ready, use another terminal for a warmup request, then repeat it for the first baseline sample:

```sh
curl --fail-with-body -sS http://127.0.0.1:18069/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-0.8B","messages":[{"role":"user","content":"List the integers from 1 to 100, separated by commas."}],"temperature":0,"repetition_penalty":1.0,"max_tokens":128,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' \
  -w '\nrequest_seconds=%{time_total}\n'
```

Save responses and server logs. Use actual `usage.completion_tokens`, which can be below 128 if EOS occurs. End-to-end output throughput is completion tokens divided by full request seconds; it includes prefill. Report server decode throughput separately. A fixed token cap is not proof of a fixed generated length.

## Capture a timeline

Warm the server before tracing. Find its PID without killing any unrelated process:

```sh
pgrep -fl crane-serve
```

Replace `<SERVER_PID>` below with the intended process ID. Record a short Instruments window and send the same request from another terminal while recording:

```sh
xcrun xctrace record --template "Metal System Trace" \
  --attach <SERVER_PID> --time-limit 15s --output crane-metal-baseline.trace
open crane-metal-baseline.trace
```

Use an available Metal/Game Performance template if the installed Xcode version names them differently. Inspect CPU submission/encoding, GPU execution, gaps, buffer copies and CPU waits together. Use Time Profiler for CPU stacks and Xcode's Metal debugger for deeper shader/counter analysis. A long CPU API call may be waiting for earlier GPU work rather than transferring data for its whole duration.

`CRANE_PROF=1` can add coarse engine spans. `CRANE_PROF_SYNC=1` forces synchronization and changes overlap, so do not use it for normal performance claims. Benchmark outside Instruments with profiling disabled.

Apple references:

- https://developer.apple.com/metal/tools/
- https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/
- https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/porting-discover/reference/gptk-and-traces.md

## Evidence needed before keeping a change

1. **One hypothesis at a time.** Record the unchanged trace and candidate trace, changed operation counts, relevant GPU timings and CPU waits. Preserve an unchanged baseline binary and exact commit/patch for each result.
2. **Correctness first.** Compare the candidate to the old path on the same device/dtype. Add focused coverage for any new dispatch guard. For attention cover GQA, strided cache views, context 1/255/256/257 and 1,023/1,024/1,025, and explicit-mask fallback. Use appropriate numerical tolerances and record maximum error; do not loosen thresholds to hide unexplained differences.
3. **End-to-end validation.** Compare logits and greedy output across multiple prompts, cache reuse and context lengths. Existing `qwen3_5_norms`, `metal_kernels` and `gdn_fused` tests are useful starting points; confirm that tests actually exercise Metal, since some device helpers skip when unavailable. Existing tests do not prove the new ordered low-precision path correct.
4. **Paired timings.** Use five paired rounds, alternating baseline/candidate order, with ten fixed prompts per version per round and identical sampling. Warm each binary, distinguish cold prefill from prefix-cache hits, preserve all responses and use actual token counts. Keep dtype, checkpoint, context capacity, occupied context, concurrency, charger and thermal conditions comparable. Report medians, per-round gains and baseline drift.
5. **Long-context attention.** Start small, then choose larger context capacity if memory permits. Verify actual prompt token counts and remaining generation capacity. `--context 2048` sets capacity; it does not create a 2,048-token input. Test occupied lengths on both sides of the native attention implementation boundary without mixing prefill and decode timing.
6. **Review scope.** Keep unrelated CUDA, scheduler, quantization and formatting changes out. Capture precise validation commands/results and limitations before proposing a separate Metal PR. No claim about all Macs follows from one machine.

Store raw results outside tracked source; do not commit checkpoint weights, binaries, large Instruments traces, credentials or machine-specific paths. End the investigation with the chosen change, evidence, correctness limits and remaining uncertainty.
