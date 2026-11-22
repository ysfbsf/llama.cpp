# llama.cpp

> [!NOTE]
> **This fork: faster Inkling 2 on CUDA.** Branch `inkling2-speed` is upstream master plus four commits that make
> Inkling 2 Mini run faster on NVIDIA GPUs: the depth states are normalized once per block, and the model's small glue
> op chains run as fused CUDA kernels (`ggml/src/ggml-cuda/inkling2.cu`). Build it the usual way (`-DGGML_CUDA=ON`);
> nothing needs to be enabled at runtime, and other models are unaffected.
>
> ![Inkling 2 Mini, upstream vs fork, RTX 5060 Ti](media/inkling2-speed-5060ti.png)
>
> RTX 5060 Ti 16 GB, Inkling-2-Mini-UD-IQ1_S, `llama-bench -fa 1 -ngl 99`, f16 KV, tokens/s:
>
> | context depth | prefill (pp512) upstream | prefill fork | decode (tg128) upstream | decode fork |
> |---:|---:|---:|---:|---:|
> | 0 | 1097 | 2193 (2.0x) | 64.5 | 88.8 (+38%) |
> | 4K | 725 | 1903 (2.6x) | 60.6 | 84.2 (+39%) |
> | 8K | 626 | 1783 (2.8x) | 59.5 | 81.8 (+38%) |
> | 16K | 494 | 1595 (3.2x) | 56.0 | 77.4 (+38%) |
> | 32K | 349 | 1320 (3.8x) | 50.6 | 69.5 (+37%) |
> | 48K | 270 | 1108 (4.1x) | 46.4 | 63.6 (+37%) |
> | 64K (`-ub 256`) | 222 | 902 (4.1x) | 42.9 | 58.8 (+37%) |
>
> ![Inkling 2 Mini, upstream vs fork, Tesla V100](media/inkling2-speed-v100.png)
>
> Tesla V100 32 GB, Inkling-2-Mini-UD-Q4_K_XL, `llama-bench -fa 1 -ngl 99`, f16 KV, tokens/s:
>
> | context depth | prefill (pp512) upstream | prefill fork | decode (tg128) upstream | decode fork |
> |---:|---:|---:|---:|---:|
> | 0 | 725 | 1185 (1.6x) | 32.0 | 50.7 (+59%) |
> | 4K | 500 | 1093 (2.2x) | 31.2 | 49.6 (+59%) |
> | 8K | 436 | 1044 (2.4x) | 30.8 | 49.0 (+59%) |
> | 16K | 349 | 943 (2.7x) | 30.0 | 47.9 (+60%) |
> | 32K | 250 | 825 (3.3x) | 28.5 | 45.7 (+60%) |
> | 48K | 194 | 725 (3.7x) | 27.2 | 43.8 (+61%) |
> | 64K | 158 | 651 (4.1x) | 25.6 | 41.4 (+62%) |
> | 96K | 116 | 533 (4.6x) | 23.1 | 38.5 (+67%) |
> | 128K (`-ub 256`) | 94 | 467 (4.9x) | 21.1 | 35.4 (+68%) |
>
> At the largest depth on each card, the default `-ub 512` runs out of memory for both builds, so that row uses `-ub 256`.
>
> Output quality matches upstream. On the RTX 5060 Ti most fused kernels reproduce the unfused results bit for bit; on
> the V100 they differ in the last bits. Perplexity KLD against upstream, next to the KLD between two upstream runs at
> different batch sizes: 5060 Ti decode 0.0018 vs 0.0027, prefill 0.0022 vs 0.0031; V100 decode 0.0025 vs 0.0022 (within
> the measurement error), prefill 0.0016 vs 0.0017. `test-backend-ops -o INKLING2_*` covers each fused chain.
>
> **AMD (user report from [@m_check1B](https://x.com/m_check1B/status/2124237027847938484), not run by us).** The fused
> kernels also build for AMD GPUs: the HIP backend
> (`-DGGML_HIP=ON`) compiles the same CUDA sources, so no changes were needed. Radeon AI PRO R9700, Inkling-2-Mini
> Q8_0, upstream and fork run back to back with identical settings, tokens/s:
>
> | context depth | prefill upstream | prefill fork | decode upstream | decode fork |
> |---:|---:|---:|---:|---:|
> | 0 | 1567 | 2378 (+52%) | 32.2 | 48.9 (+52%) |
> | 8K | 685 | 1769 (2.6x) | - | - |
> | 32K | 344 | 1190 (3.5x) | 30.2 | 47.4 (+57%) |
>
> They report that all 20,882 `test-backend-ops` cases pass, including the `INKLING2_*` ones, and that spot-check
> prompts came back correct.

> [!WARNING]
> **12 GB cards: keep the loop layers on the GPU.** Inkling-2-Mini Q4_K (16.5 GB) does not fit in 12 GB. Upstream, when experts of the loop layers (`blk.6`-`blk.17`, run 3x per token) are on the CPU, any prompt longer than ~96 tokens comes out as all `?` (upstream [#32755](https://github.com/ggml-org/llama.cpp/issues/32755), [#32778](https://github.com/ggml-org/llama.cpp/issues/32778)), and tool calls fail with `Unexpected empty grammar stack after accepting piece: ? (30)`. Cause: for large batches the scheduler offloads these CPU weights to the GPU (op offload), but it copies a weight only in the first split that uses it, so the later loop passes read a stale copy. `--no-op-offload` avoids it. This branch fixes it in `ggml-backend.cpp` (the weight is copied again in each split that uses it). Tested on an RTX 5070: auto-fit, `--n-cpu-moe 12` and loop experts on the CPU all give correct output now. Upstream [#32777](https://github.com/ggml-org/llama.cpp/pull/32777) alone does not fix it.
>
> Loop experts on the CPU are still slow. For speed, put every layer on the GPU and send only the experts of the single-pass layers to the CPU:
>
> ```sh
> llama-server -m Inkling-2-Mini-Q4_K.gguf --jinja -fa on -c 32768 -np 1 -ngl 99 \
>   -ot 'blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU'
> ```
>
> Do not use `-ngl 99` without `-ot`: the model overflows VRAM and runs at ~1 t/s. RTX 5070 12 GB, Ryzen 5 7600X, DDR5-6000, Q4_K, this command, real `llama-server` requests, tokens/s:
>
> | | prefill (~3.5K prompt) | decode |
> |---|---:|---:|
> | upstream b12620 | 119 | 36 |
> | this fork | 132 | 43 |
>
> On this setup `-np 2` cut decode to ~23 t/s, and `-ub 2048` cut prefill to ~40 t/s.

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
