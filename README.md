<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

> [!IMPORTANT]
> **This is an unofficial fork** ([shiosalt/Strata](https://github.com/shiosalt/Strata)) of
> [Niko1221/Strata](https://github.com/Niko1221/Strata). It adds one model,
> **NVIDIA-Nemotron-3.5-Super-VL-120B-A12B**, on an engine of its own. The Strata maintainers did not write
> or review it: report problems with it [here](https://github.com/shiosalt/Strata/issues), not upstream.
> Everything else in this README is upstream's, and the installers and release links below are upstream's
> (they do not contain the Nemotron engine). See [Models](#nemotron-models).

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

<a id="nemotron-models"></a>

### Nemotron 3.5 Super (this fork)

[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF): 120B parameters, 12B active per token, reads pictures. This fork runs Unsloth's **UD-IQ3_XXS** (59.8 GB + a 1.7 GB picture reader) on an engine of its own, in the Terminal. Text, thinking, tool calls, pictures, MTP drafts and 128K / 256K context work. It answers one request at a time.

Measured on one PC only (2 × RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB RAM, Windows 11, 128K context):

| | Speed |
| --- | --- |
| Generating, a short request (temperature 1.0, MTP 6) | 24-28 tokens/s |
| Generating, 72K tokens into a conversation | 12.7-15.4 tokens/s |
| Reading a 36,529-token prompt | 276 tokens/s (132 s) |

How to install it: [Install](#nemotron-install). The weights are NVIDIA's, under the license on its model card; this repository contains none. Details and all measurements (in Japanese): [docs/NEMOTRON35.md](docs/NEMOTRON35.md).

**With the original Strata features, these models can also be used:**

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (ISTA-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |
| ISTA-DAS Coder (half of the experts removed) | 191 GB | about 197 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="nemotron-install"></a>

> [!NOTE]
> **This fork was checked on Windows only. Nemotron 3.5 Super runs on NVIDIA GPUs (CUDA) only.** The desktop app and upstream's installers below do not run it: use this repository's Terminal.

**Nemotron 3.5 Super with the ready-made engine (needs Python 3.10+, Git and an NVIDIA driver 580 or newer):**

1. Get this repository: `git clone https://github.com/shiosalt/Strata.git`, then `cd Strata`.
2. Download `strata-nemotron-0.1.0-windows-x64-cuda13.zip` from [Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) and unzip it into the repository folder (it makes `engine-nemotron\`).
3. Install the Python packages:

   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8
   ```

4. Download the model (or point `--model-path` at files you have) and serve it:

   ```bat
   .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
   ```

   `.venv\Scripts\python.exe strata.py launch --stop` stops it. The installed `strata` command is upstream's and does not know this model.

To build the engine yourself instead (Visual Studio 2022 Build Tools with "Desktop development with C++" and the CUDA Toolkit 12.8 or newer; 30 minutes or more), run `powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision` in place of steps 2 and 3.

The rest of this section is upstream's, for its own models.

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

> [!NOTE]
> **This fork was checked on Windows only. Nemotron 3.5 Super runs on NVIDIA GPUs (CUDA) only.**

**Nemotron 3.5 Super:** NVIDIA GPUs with about 30 GB of VRAM in total, 64 GB of RAM or more (it warns, but does not refuse, below that) and 62 GB of disk for the model. Measured only on 2 × RTX 5060 Ti 16 GB with 96 GB of RAM. The ready-made engine carries code for RTX 30 (sm_86), RTX 40 (sm_89) and RTX 50 (sm_120) cards; only RTX 50 was tried. Linux and smaller PCs were not tried.

The table below is upstream's, for its own models.

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** These links support **Niko ([@Niko1221](https://github.com/Niko1221)), the author of Strata**, not this fork. Thank you, Niko, for writing Strata and sharing it as open source: this fork is built on his work.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

