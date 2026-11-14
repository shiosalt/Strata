# NVIDIA-Nemotron-3.5-Super-VL-120B-A12B を Strata で動かす

この fork は Nemotron-3.5-Super-VL(GGUF の系統名 `nemotron_h_moe`)を Strata 自身のエンジン(`src/nemotron`)で動かします。
expert キャッシュ(VRAM のスロット)、pinned RAM の expert と CPU の expert プール、MTP のドラフト(温度 > 0 でも)、prompt lookup、
2 GPU の層分割、画像、会話のパーキングとセッションファイルに対応しています。進み具合と測った数字の全部は
[NEMOTRON35_PROGRESS.md](NEMOTRON35_PROGRESS.md) にあります。

実モデルで確認済み(2026-11-10〜13、下の PC): チャット、思考、ツール呼び出し、画像、会話の続きの再利用、MTP、128K / 256K の起動、
30,514 トークンの長文からの検索、`strata.py launch --family nemotron35` での導入から起動まで。

## 測った速さ

RTX 5060 Ti 16GB × 2(sm_120。GPU 0 は PCIe Gen3 x4、GPU 1 は Gen4 x8)、Ryzen 7 5700X、RAM 96GB、Windows。UD-IQ3_XXS、128K、
チャンク 4096。今の設定(MTP 6、`--mtp-slots 64`)での数字(2026-11-13〜14、各 1〜4 回):

| 項目 | 速さ |
| --- | --- |
| 生成(短い依頼「九九の表を作成して」、思考なし、温度 1.0) | 24〜28 tok/s(1 回目は 20〜22) |
| 生成(72K トークンの会話の続き、温度 1.0) | 12.7 / 15.4 tok/s |
| prompt 36,529 トークン(2 GPU の流れ作業) | 132 秒(276 tok/s) |
| サーバーの起動(読み込み済みのファイル) | 約 70〜80 秒 |

細かい内訳と、どの変更で何が変わったかは [NEMOTRON35_PROGRESS.md](NEMOTRON35_PROGRESS.md) にあります。下の表は最初の版
(2026-11-10、MTP 3、流れ作業とアテンションの書き直しの前)の数字です。

| 項目 | 速さ |
| --- | --- |
| 起動 | 約 2.5 分(`strata launch`、expert 49.2 GiB を pinned RAM にコピーする 55 秒を含む) |
| 生成(温度 1.0、MTP 3) | 14.7 tok/s(MTP なしは 10.5) |
| 生成(greedy) | 10.2 tok/s |
| 生成(コードの書き直し、prompt lookup) | 17.8 tok/s |
| prompt 29 トークン | 2.2 秒 |
| prompt 721 トークン | 108.5 tok/s |
| prompt 3,218 トークン | 166.8 tok/s |
| prompt 30,514 トークン | 159 秒(192 tok/s。中ほどの暗号を正しく答えた) |
| 会話の続き(別の会話を挟んだあと、1,547 トークン) | 1.7 秒(パーキングなしは 12.5 秒) |

生成の速さは同じサーバーで何件か処理すると上がります(使われる expert を VRAM に入れ替えるため。前回の expert プロファイルからも
始めます)。上の数字は起動直後のものではなく、その状態のものです。

## 導入して起動する(Terminal)

エンジンは本家のリリースのパッケージに入っていません。この fork の
[Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) にあるビルド済みのエンジン
(`strata-nemotron-0.1.0-windows-x64-cuda13.zip`、Windows x64、NVIDIA ドライバー 580 以降、AVX2 の CPU)をこのチェックアウトに展開する
(`engine-nemotron\` ができる)か、自分でビルドします(下の「ビルド」)。ビルド済みのエンジンには RTX 30 / 40 / 50(sm_86 / sm_89 /
sm_120)向けのコードが入っていますが、試したのは RTX 50 だけです。作り方は `tools\nemotron\package.ps1` にあります。
展開した場合は Python のパッケージを入れてから(`python -m venv .venv` のあと
`.venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8`)、このチェックアウトの Terminal(`strata.py`)で導入して
起動します。インストーラーで入れた `strata` コマンドは本家のもので、
このモデルを知りません。

```bat
.venv\Scripts\python.exe strata.py launch --plan --family nemotron35 --model UD-IQ3_XXS
.venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
.venv\Scripts\python.exe strata.py launch --stop
```

- モデルのファイル(3 シャード 59.8 GB と画像リーダー 1.7 GB)は固定のリビジョンからダウンロードし、サイズと SHA-256 を確かめます。
  置き場所は `<データフォルダ>\models\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF\UD-IQ3_XXS\`。
- 手元のファイルを使うとき: `strata.py launch --model-path <folder>\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`(サイズのフォルダ、その上の
  フォルダ、シャード 1 つのどれでも)。何もコピーしません。
- エンジンは次の順に探します: 環境変数 `STRATA_NEMOTRON_ENGINE`、`<インストール先>\engine-nemotron\strata.exe`、
  `<チェックアウト>\build\nemotron-cuda\strata.exe`、`<チェックアウト>\engine-nemotron\strata.exe`(ビルド済みのエンジン)。無いときはダウンロードの前に止まり、その旨を 1 行で言います。
- 画像リーダーは `STRATA_NEMOTRON_VISION`、`<インストール先>\engine-nemotron\strata-vision.exe`、`build\vision-cpu`(`--vision gpu` なら
  `build\vision-cuda`)、`<チェックアウト>\engine-nemotron\strata-vision.exe` の順。無いときは画像なしで起動します。
- 使えるオプション: `--context 128k|256k`、`--vision off|cpu|gpu`(既定 cpu)、`--gpu 0,1`、`--port`、`--access`、`--api-key-env`。
  `--parallel N` は使いません(このエンジンは 1 件ずつ答え、残りは待ちます)。`--kv` も使いません(K / V は BF16)。
  チューニング(`--tune`)の代わりに下の `calibrate.py` があります。
- setup.py からも同じことができます: `python setup.py --family nemotron35 --model UD-IQ3_XXS --no-start`(`--gguf-dir` で手元のファイル)。
  `START-HERE.bat` / `SETUP.bat` の「Step 2: your choices」のメニューにも出ます(NVIDIA の GPU のときだけ。エンジンが見つかったかを
  行の末尾に書きます)。step 1 で選んだカードがすべて使われます。
  設定は `strata-nemotron35-ud-iq3_xxs.json`。トークナイザと expert プロファイルは `<データフォルダ>\nemotron\nemotron35-ud-iq3_xxs\`。
- アプリの「手元のモデルを追加」も `nemotron_h_moe` の GGUF を読めます(層ごとのヘッド数の配列に対応)。ただしアプリが同梱する
  エンジンはこのモデルを動かせないので、読み込みでエラーになります。今は Terminal を使ってください。

どこで動くか: NVIDIA の GPU(CUDA)だけ。測ったのは 16 GB × 2 枚と RAM 96 GB です。GPU の合計が 30 GB 未満、RAM が 64 GB 未満の PC では
警告を出します(断りはしません)。小さい PC では試していません。

## 自分の PC で設定を測る

```bat
python tools\nemotron\calibrate.py --config strata-nemotron35-ud-iq3_xxs.json
```

MTP のドラフト数(既定 0, 2, 3, 4)と prompt のチャンク(2048, 4096)を 1 つずつ起動して測り(温度 1.0)、今の値より 3% 以上速い
ものだけを設定に書きます。元の設定は `.before-calibrate.json` として残ります。起動 1 回が約 2.5 分なので、既定では 20 分前後
かかります。`--dry-run` で何を測るかだけ表示します。PCIe の分担はエンジンが起動のたびに測るので、ここでは測りません。

## ビルド

必要なもの: Windows、Visual Studio 2022(Build Tools でよい。「C++ によるデスクトップ開発」)、CUDA Toolkit 12.8 以降
(13.1 で確認)、Python 3.10 以降、Git(画像リーダーを作るとき)。

```bat
powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision
```

このスクリプトは Visual Studio と CUDA を探し、`.venv` に CMake・Ninja・サーバーと Terminal のパッケージ(固定の版)を入れ、
この PC の GPU の世代(`nvidia-smi` の compute capability)向けに `build\nemotron-cuda` を構成してビルドします。最後に
`strata.exe --families` が `nemotron_h_moe` を含むことを確かめます。初回は 30 分以上かかることがあります。ログは
`build\build-nemotron.log`。

- `-Vision`: 画像を読む strata-vision(CPU 版、VRAM を使わない)も作ります。固定の llama.cpp(3cf03257)はこの mmproj の形式
  (`nemotron_omni`)を知らないので、新しい llama.cpp(コミット aa75ff32)を `build\llama.cpp-vision` に取ってきて、それに対して
  ビルドします。エンジンの ggml は固定のままです。`-VisionGpu` は GPU 版(約 1.5 GB の VRAM)。
- `-Tests`: Nemotron のテスト(`nemotron_kernels_parity`、`nemotron_engine_parity`、`nemotron_cpu_experts_test`)も作ります。
- `-Arch 120`: GPU の世代を指定します。確かめたのは sm_120(RTX 50 系)だけです。
- Linux と Mac のビルドは試していません(エンジンは CUDA だけ)。

## Terminal を使わずに設定を作る(開発用)

先頭シャード(メタデータだけ、約 8 MB)を読んで、トークナイザ・チャットテンプレート・実行設定を `build/nemotron-run/` に書きます。
重みは読みません。中身は `strata.py launch` が書くものと同じです(どちらも `strata_control/nemotron35.py`)。

```bat
.venv\Scripts\python.exe tools\nemotron\make_config.py --model-dir <data folder>\models\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF\UD-IQ3_XXS --context 128k
.venv\Scripts\python.exe serve\server.py --engine strata --config build\nemotron-run\strata-nemotron-128k.json --port 8080
```

主なオプション: `--mtp N`(0〜7、既定 6。温度 1.0 で 3 は 15 tok/s、6 は 21 tok/s、2026-11-13)、`--chunk N`(prompt を何行ずつ読むか、既定 4096。1024 だと 3,218 トークンの prompt が
93 tok/s、4096 で 156 tok/s)、`--gpu 0,1`、`--vision cpu|gpu|off`、`--vision-max-tokens N`(既定 3328。この mmproj では llama.cpp が
画像を上限まで縮めないので、モデル自身の範囲 256〜3328 トークンの上端にしてあります)。

API は OpenAI 互換(`/v1/chat/completions`)と Anthropic 互換(`/v1/messages`)。モデル名は `nemotron-3.5-super-vl-120b-a12b`。

## エンジンのオプション(設定の `args` に足すもの)

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `--no-pin` | pin する | expert を pinned RAM にコピーしない(RAM が少ない PC 用。遅くなる) |
| `--pin-gib N` | Windows では 36 | ページ固定する expert の上限(GiB)。速いリンクの GPU の層から順に固定し、残りはファイルの mapping から CPU が読む(その層は PCIe 分担なし)。Windows ではおよそ 40 GiB を超えると固定に失敗するため |
| `--mtp-slots N` | 0(設定ファイルは 64) | MTP 層の expert を VRAM に置く数。同じ GPU の他の層のスロットから取る |
| `--no-gpu-sample` | GPU で選ぶ | サンプリングの上位 k 個を GPU で選ばず、全ロジットを CPU にコピーする(`STRATA_NEMO_GPU_SAMPLE=0` と同じ) |
| `--bench-cpu` | — | 読み込んだあと、RAM の読み出し速度と CPU の expert カーネルの速度(型ごと)を測って終わる(サーバーは起動しない) |

環境変数で元の動きに戻せるもの: `STRATA_NEMO_PIN_ALLOC=1`(expert を cudaHostAlloc で固定)、`STRATA_NEMO_CPU_KERNELS=0`(Q2_0 も
ggml の内積)、`STRATA_NEMO_GPU_SAMPLE=0`(全ロジットを CPU へ)、`STRATA_NEMO_ATTN_WIN=0`(decode のアテンションを行ごとの
以前のカーネルで)、`STRATA_NEMO_CPU_ROWS_ALL=1`(prompt の CPU の expert の結果を、計算した行だけでなく全行 GPU へ送る)。調査用: `STRATA_NEMO_ROUTE_DUMP=<file>`(リクエストのたびに、
decode の層と expert ごとの routing 回数と、VRAM にあるかを CSV に書く)。
| `--pcie-frac F` | 起動時に測り、待ち時間で調整 | decode で PCIe から読む expert の割合を固定する(全 GPU に同じ値。GPU ごとには環境変数 `STRATA_NEMO_PCIE_FRAC=f0,f1`) |
| `--no-prefill-pipe` | 流れ作業にする | 2 GPU で prompt の複数のチャンクを流れ作業にしない(GPU 1 が前のチャンクを読む間に GPU 0 が次のチャンクを読む、をやめる。`STRATA_NEMO_PIPE=0` と同じ) |
| `--no-share-adapt` | 調整する | PCIe の割合を、CPU の後に GPU を待った時間で調整しない(`STRATA_NEMO_SHARE_ADAPT=0` と同じ) |
| `--pool-cores physical\|LIST` | 固定しない | CPU の expert のスレッドを固定する(この PC では遅くなった: 14.4 → 13.0 tok/s) |
| `--no-lookup` | 使う | prompt lookup の下書きを使わない |
| `--expert-profile FILE` / `--expert-profile-save-every MIN` | 設定に書く / 10 分 | 使われた expert の記録の読み込みと保存の間隔 |
| `--prompt-cache N` / `--prompt-cache-every N` / `--conversation-cache-mib N` / `--conversation-cache-min-tokens N` | 6 / 16384 / 8192 / 1024 | チェックポイントの数と間隔、会話のパーキング(RAM の上限 MiB、最小のトークン数) |

## まだ無いもの

- ビルド済みのエンジンの配布(リリースのパッケージに入っていない。ソースからビルドします)。アプリからの導入と起動。
- 複数のリクエストを同時に生成すること(バッチのスロット)。今は 1 件ずつです。
- CUDA graph。どの expert を GPU と CPU に分けるかを層ごとにホストが決めるため、グラフにできる区間が Mamba のブロック 1 つ程度に
  なり、起動の回数はまとめたカーネルで既に減らしてあります。
- logprobs と logit_bias(エラーで断ります)。
- llama.cpp とのロジットの照合はしていません(出力の内容が正しいことは下の確認で見ています)。
- 128K いっぱいの prompt は測っていません(30K で 192 tok/s なので 10 分を超えるはずです)。

## 動作確認の手順

2026-11-10 にすべて実行して通った手順です。やり直すときに使ってください。

1. **起動と短い生成**(数分)。モデルを読み込み、日本の首都を聞いて 48 トークン生成します。

   ```bat
   .venv\Scripts\python.exe tools\nemotron\smoke.py --n 48
   ```

   見てほしい所: 最後の `text:` が意味の通る答えか。stderr の `strata-nemotron:` で始まる行(読み込み時間、GPU の割り当て、
   スロット数、prompt / 生成の tok/s)。

2. **非有限値の検査つき**(1 が変な文字列だった場合だけ):

   ```bat
   .venv\Scripts\python.exe tools\nemotron\smoke.py --n 8 --check
   ```

   `CHECK block ...` の行が出たら、その行を送ってください。

3. **サーバー(128K)**: 上の起動コマンドのあと、別のターミナルで:

   ```bat
   curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about autumn.\"}],\"max_tokens\":400}"
   ```

   ツール呼び出しも 1 回(リクエストは `build\nemotron-run\tool-call.json`。返事の `finish_reason` が `tool_calls` になるか):

   ```bat
   curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @build\nemotron-run\tool-call.json
   ```

   止めてから 256K の設定でも起動し、起動のログ(VRAM、スロット数)だけ送ってください。

4. **画像**: サーバー(CPU の画像リーダー)に画像を 1 枚送ります。手元のファイルは data URL(base64)で送れます(PowerShell):

   ```powershell
   $b = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\picture.jpg"))
   $body = @{ model = "m"; max_tokens = 2000; messages = @(@{ role = "user"; content = @(
       @{ type = "text"; text = "What is in this picture?" },
       @{ type = "image_url"; image_url = @{ url = "data:image/jpeg;base64,$b" } }) }) } | ConvertTo-Json -Depth 8
   Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType "application/json" -Body $body |
       ConvertTo-Json -Depth 8
   ```

   見てほしい所: ログの `strata-vision:` の行(読み込み時間)、画像トークン数とエンコード時間、答えが画像に合っているか。
   (思考に max_tokens を使い切ることがあるので、400 より大きめに。)

5. **llama.cpp との照合**(式の読み取りの最終確認、任意): 同じプロンプトで llama.cpp(master)の上位 10 個のロジットと、
   `llama-tokenize` の ID 列。
