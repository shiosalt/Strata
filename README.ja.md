<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">あなた自身のパソコンで、高性能な AI を。</h3>
<p align="center">チャット、コーディング、画像の処理を、すべてローカルで。</p>

<p align="center">
  <a href="#モデル">モデル</a> · <a href="#インストール"><b>インストール</b></a> ·
  <a href="#ハードウェア">ハードウェア</a> · <a href="#ツール">ツール</a> ·
  <a href="#ヘルプ">ヘルプ</a> · <a href="#strata-を支援する">支援</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Strata をダウンロード"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT ライセンス"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="GitHub Sponsors で Strata を支援する"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Buy Me a Coffee で Strata を支援する"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

> [!IMPORTANT]
> **これは非公式の fork です**([shiosalt/Strata](https://github.com/shiosalt/Strata)。元は [Niko1221/Strata](https://github.com/Niko1221/Strata))。**NVIDIA-Nemotron-3.5-Super-VL-120B-A12B** を専用のエンジンで動かす機能を足しています。この部分は Strata の開発者が書いたものでも確認したものでもありません。問題は本家ではなく[こちら](https://github.com/shiosalt/Strata/issues)に報告してください。この README のほかの部分は本家のもので、下のインストーラーとリリースのリンクも本家のものです(Nemotron のエンジンは入っていません)。[モデル](#nemotron-models)を見てください。

**無料でオープンソース。自分の環境で動かせます。** デスクトップで、またはお好みのコーディングツールと一緒に Strata を使えます。アカウントは不要です。

## モデル

<a id="nemotron-models"></a>

### Nemotron 3.5 Super(この fork)

[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF): パラメータ 1,200 億(1 トークンあたり 120 億が動く)、画像も読めます。この fork は Unsloth の **UD-IQ3_XXS**(59.8 GB と画像リーダー 1.7 GB)を専用のエンジンで、Terminal から動かします。文章、思考、ツール呼び出し、画像、MTP の下書き、128K / 256K のコンテキストに対応しています。リクエストは 1 件ずつ処理します。

測った PC は 1 台だけです(RTX 5060 Ti 16 GB × 2、Ryzen 7 5700X、RAM 96 GB、Windows 11、128K):

| | 速さ |
| --- | --- |
| 生成(短い依頼、温度 1.0、MTP 6) | 24-28 tokens/s |
| 生成(72K トークンの会話の続き) | 12.7-15.4 tokens/s |
| 36,529 トークンの prompt の読み込み | 276 tokens/s (132 s) |

導入方法は[インストール](#nemotron-install)にあります。モデルの重みは NVIDIA のもので、そのモデルカードのライセンスに従います(このリポジトリには入っていません)。詳しい説明と測定の全部: [docs/NEMOTRON35.md](docs/NEMOTRON35.md)。

**オリジナルの Strata の機能で、下記のモデルも利用できます:**

まず **ISTA-DASLab** をおすすめします。次に **Unsloth** などの提供元です。メモリに収まる互換モデルがある場合に選べます。

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| 提供元 | 選択肢 |
| --- | --- |
| **ISTA-DASLab · おすすめ** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S。[Coder: IQ1\_M](docs/MODELS.md#coder)（コーディング向け）。 |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)。メモリ使用量は少ないですが、動作は遅くなります。 |
| **その他の選択肢** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[サイズ、必要なメモリ、互換性を比べる](docs/MODELS.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0。現在、このモデルで文書化されているパックです。

[モデルガイドと互換性](docs/QWEN36.md)。

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

**デスクトップアプリ**と**ターミナル**で使えます。メモリに収まらない分は SSD から読み込むため、空きディスク容量がたくさん必要です。

| セット（ISTA-DAS） | ダウンロード | 必要な空き容量 |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | 約245 GB |
| 2.00 bpw | 257 GB | 約265 GB |
| 2.25 bpw | 274 GB | 約280 GB |

NVIDIA RTX 30・40・50 シリーズ（Windows と Linux）、AMD Radeon AI PRO R9700（Linux）、AMD Ryzen AI Max — Strix Halo（Windows）、メモリ 64 GB 以上の Apple シリコン搭載 Mac（[Mac](docs/DEEPSEEK_MAC.md)）で動きます。セットアップがファイルと DeepSeek エンジンをダウンロードし、モデルを起動します。

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Strata 0.2.5 で 2.00 bpw のセットを使って測定しました。「出力」は回答の速さ、「プロンプト」は Strata が入力を読み込む速さです。

| ハードウェア | 出力 | プロンプト |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4〜6 トークン/秒 | 約 110 トークン/秒 (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9〜11 トークン/秒 | 約 155 トークン/秒 (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21〜25 トークン/秒 | 320〜360 トークン/秒 (8K) |
| 4x Radeon AI PRO R9700 | 74〜77 トークン/秒 | 400〜450 トークン/秒 (8K) |

[メモリ、ディスク、詳細](docs/DEEPSEEK41.md) · [リリースノート](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

どれが動くかわからない場合は、[お使いのパソコンに合うモデルを選ぶ](docs/MODELS.md#pick-by-memory)。

## インストール

<a id="nemotron-install"></a>

> [!NOTE]
> **この fork は Windows でのみ検証しています。Nemotron 3.5 Super は NVIDIA の GPU(CUDA)でのみ動きます。** デスクトップアプリと下の本家のインストーラーでは動きません。このリポジトリの Terminal を使ってください。

**Nemotron 3.5 Super をビルド済みのエンジンで使う(Python 3.10 以降、Git、NVIDIA ドライバー 580 以降が必要):**

1. このリポジトリを取得します: `git clone https://github.com/shiosalt/Strata.git` のあと `cd Strata`。
2. [Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) から `strata-nemotron-0.1.0-windows-x64-cuda13.zip` をダウンロードし、リポジトリのフォルダーに展開します(`engine-nemotron\` ができます)。
3. Python のパッケージを入れます:

   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8
   ```

4. モデルをダウンロードして(手元のファイルがあれば `--model-path` で指定)起動します:

   ```bat
   .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
   ```

   止めるときは `.venv\Scripts\python.exe strata.py launch --stop`。インストール済みの `strata` コマンドは本家のもので、このモデルを知りません。

エンジンを自分でビルドする場合は(Visual Studio 2022 Build Tools の「C++ によるデスクトップ開発」と CUDA Toolkit 12.8 以降。30 分以上かかります)、手順 2 と 3 の代わりに `powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision` を実行します。

この節のここから下は本家の説明で、本家のモデル向けです。

<a id="option-a-the-strata-app"></a>

### デスクトップアプリ

**[Windows、Linux、Mac 用の Strata をダウンロード →](https://github.com/Niko1221/Strata/releases/latest)**

アプリをインストールし、モデルを選んで、チャットを始めましょう。インストーラーはシステムごとに1つです。初回起動時に、アプリがお使いのグラフィックスカード用のエンジンをダウンロードします。[インストーラーのヘルプ](docs/INSTALL.md#the-app)。

<sub>ファイル: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="チャットとライブのコードプレビューを並べて表示した Strata デスクトップアプリ">
  <br><sub>チャット、ファイル、ライブプレビューを1つのウィンドウで。</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### ターミナルアプリ

ターミナルのほうが好みですか？ Strata をインストールして、**`strata launch`** を実行します。

<details open>
<summary><b>インストールコマンド</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Windows では、インストール後に新しいターミナルを開きます。そのあと:

```sh
strata launch
```

画面の案内に従って、モデルをダウンロードして起動します。[ターミナルガイド](docs/TERMINAL_LAUNCH.md)。

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="モデルの動作、速度、メモリ、最近のリクエストを表示する Strata ターミナルアプリ">
  <br><sub>動作中のモデルとハードウェアの状態を確認できます。</sub>
</p>

<a id="build-from-source"></a>

[ソースからビルド](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [インストールガイド（全文）](docs/INSTALL.md)

<a id="what-you-need"></a>

## ハードウェア

> [!NOTE]
> **この fork は Windows でのみ検証しています。Nemotron 3.5 Super は NVIDIA の GPU(CUDA)でのみ動きます。**

**Nemotron 3.5 Super:** NVIDIA の GPU で VRAM が合計約 30 GB、RAM 64 GB 以上(下回ると警告しますが、止めはしません)、モデル用にディスク 62 GB。測ったのは RTX 5060 Ti 16 GB × 2、RAM 96 GB の PC だけです。ビルド済みのエンジンは RTX 30(sm_86)、RTX 40(sm_89)、RTX 50(sm_120)向けのコードを含みますが、試したのは RTX 50 だけです。Linux と小さい PC は試していません。

下の表は本家のもので、本家のモデル向けです。

| 対応ハードウェア | ガイド |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 シリーズ** | [NVIDIA のセットアップ](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900、RX 9060 / 9070、AI PRO R9700** | [AMD のセットアップ](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · 試験的** | [RX 6000 のセットアップ](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel のセットアップ](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 以降** | [Mac のセットアップ](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo のセットアップ](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## ツール

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**。OpenAI 互換と Anthropic 互換の API を通じて、ほかのツールも使えます。

**`strata connect`** を実行すると、お使いのツールの設定方法が表示されます。[接続ガイド](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)。

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## ヘルプ

[すべてのガイド](docs) · [トラブルシューティング](docs/TROUBLESHOOTING.md) · [不具合を報告](https://github.com/Niko1221/Strata/issues/new/choose) ·
[ベンチマーク](docs/COMMUNITY_BENCHMARKS.md) · [仕組み](docs/HOW_IT_WORKS.md) · [貢献する](CONTRIBUTING.md)

## Strata を支援する

**ローカル AI をみんなに開かれたものに。** 下のリンクは、この fork ではなく **Strata の作者 Niko さん([@Niko1221](https://github.com/Niko1221))** への支援です。Strata を作り、オープンソースで公開してくださった Niko さんに感謝します。この fork は Niko さんの仕事の上に成り立っています。

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: Strata を支援する" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: Strata を支援する" height="48"></a>

[MIT ライセンス](LICENSE) · [クレジットとモデルのライセンス](docs/HOW_IT_WORKS.md#credits) · [プライバシー](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>AI ツールと自動リーダー向け: <a href="AGENTS.md">技術リファレンス</a> · <a href="docs/AI_SETUP.md">セットアップ手順</a></sub>
