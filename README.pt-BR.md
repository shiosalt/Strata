<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">IA poderosa no seu próprio computador.</h3>
<p align="center">Converse, programe e trabalhe com imagens, localmente.</p>

<p align="center">
  <a href="#modelos">Modelos</a> · <a href="#instalação"><b>Instalação</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#ferramentas">Ferramentas</a> ·
  <a href="#ajuda">Ajuda</a> · <a href="#apoie-o-strata">Apoiar</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Baixar o Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licença MIT"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Apoie o Strata no GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Apoie o Strata no Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b> · <a href="README.pl.md">Polski</a></sub></p>

> [!IMPORTANT]
> **Este é um fork não oficial** ([shiosalt/Strata](https://github.com/shiosalt/Strata)) de [Niko1221/Strata](https://github.com/Niko1221/Strata). Ele adiciona o **NVIDIA-Nemotron-3.5-Super-VL-120B-A12B** com um motor próprio. Os desenvolvedores do Strata não escreveram nem revisaram esta parte: relate problemas com ela [aqui](https://github.com/shiosalt/Strata/issues), não no projeto original. O resto deste README é do projeto original, assim como os instaladores e os links de versões abaixo (eles não trazem o motor do Nemotron). Veja [Modelos](#nemotron-models).

**Gratuito, de código aberto e seu para executar.** Use o Strata no desktop ou com suas ferramentas de programação favoritas. Não precisa de conta.

## Modelos

<a id="nemotron-models"></a>

### Nemotron 3.5 Super (este fork)

[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF): 120 bilhões de parâmetros, 12 bilhões ativos por token, lê imagens. Este fork roda o **UD-IQ3_XXS** da Unsloth (59,8 GB mais um leitor de imagens de 1,7 GB) com um motor próprio, no Terminal. Texto, raciocínio, chamadas de ferramentas, imagens, rascunhos MTP e contexto de 128K / 256K funcionam. Ele atende uma requisição por vez.

Medido em um único PC (2 × RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB de RAM, Windows 11, contexto de 128K):

| | Velocidade |
| --- | --- |
| Geração, pedido curto (temperatura 1,0, MTP 6) | 24-28 tokens/s |
| Geração, a 72K tokens de uma conversa | 12.7-15.4 tokens/s |
| Leitura de um prompt de 36.529 tokens | 276 tokens/s (132 s) |

Como instalar: [Instalação](#nemotron-install). Os pesos são da NVIDIA, sob a licença do cartão do modelo; este repositório não traz nenhum. Detalhes e todas as medições (em japonês): [docs/NEMOTRON35.md](docs/NEMOTRON35.md).

**Com os recursos originais do Strata, também é possível usar estes modelos:**

Recomendamos **primeiro a ISTA-DASLab**, depois a **Unsloth** e outros provedores, quando um modelo compatível cabe na memória.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provedor | Opções |
| --- | --- |
| **ISTA-DASLab · recomendado** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) para programar. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Usa menos memória, mas é mais lento. |
| **Outras opções** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare tamanhos, necessidades de memória e compatibilidade](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Esses são os pacotes documentados atualmente para este modelo.

[Guia do modelo e compatibilidade](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

No **aplicativo de desktop** e no **Terminal**. O que não cabe na memória é lido do seu SSD, por isso é preciso muito espaço livre em disco.

| Conjunto (ISTA-DAS) | Download | Espaço livre necessário |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | cerca de 245 GB |
| 2.00 bpw | 257 GB | cerca de 265 GB |
| 2.25 bpw | 274 GB | cerca de 280 GB |

Roda em NVIDIA RTX 30, 40 e 50 (Windows e Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo (Windows) e Macs com Apple Silicon e 64 GB ou mais ([Mac](docs/DEEPSEEK_MAC.md)). A instalação baixa os arquivos e o motor do DeepSeek e depois inicia o modelo.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Medido com o conjunto de 2.00 bpw no Strata 0.2.5. “Saída” é a velocidade da resposta; “Prompt” é a rapidez com que o Strata lê o seu texto.

| Hardware | Saída | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB de RAM | 4 a 6 tokens/s | cerca de 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB de RAM | 9 a 11 tokens/s | cerca de 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB de RAM | 21 a 25 tokens/s | 320 a 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 a 77 tokens/s | 400 a 450 tokens/s (8K) |

[Memória, disco e detalhes](docs/DEEPSEEK41.md) · [Notas da versão](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Não sabe o que cabe? [Escolha um modelo para o seu computador](docs/MODELS.md#pick-by-memory).

## Instalação

<a id="nemotron-install"></a>

> [!NOTE]
> **Este fork foi verificado apenas no Windows. O Nemotron 3.5 Super roda apenas em GPUs NVIDIA (CUDA).** O aplicativo de desktop e os instaladores originais abaixo não o rodam: use o Terminal deste repositório.

**Nemotron 3.5 Super com o motor pronto (precisa de Python 3.10+, Git e um driver NVIDIA 580 ou mais novo):**

1. Baixe este repositório: `git clone https://github.com/shiosalt/Strata.git` e depois `cd Strata`.
2. Baixe `strata-nemotron-0.1.0-windows-x64-cuda13.zip` em [Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) e descompacte na pasta do repositório (isso cria `engine-nemotron\`).
3. Instale os pacotes Python:

   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8
   ```

4. Baixe o modelo (ou indique seus arquivos com `--model-path`) e inicie:

   ```bat
   .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
   ```

   `.venv\Scripts\python.exe strata.py launch --stop` para o servidor. O comando `strata` instalado é do projeto original e não conhece este modelo.

Para compilar o motor você mesmo (Visual Studio 2022 Build Tools com "Desenvolvimento para desktop com C++" e CUDA Toolkit 12.8 ou mais novo; 30 minutos ou mais), rode `powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision` no lugar das etapas 2 e 3.

O resto desta seção é do projeto original, para os modelos dele.

<a id="option-a-the-strata-app"></a>

### Aplicativo de desktop

**[Baixar o Strata para Windows, Linux ou Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Instale o aplicativo, escolha um modelo e comece a conversar. Há um instalador por sistema. Na primeira vez que abre, o aplicativo baixa o motor da sua placa de vídeo. [Ajuda com o instalador](docs/INSTALL.md#the-app).

<sub>Arquivos: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Aplicativo de desktop do Strata com um chat e uma prévia do código ao vivo lado a lado">
  <br><sub>Chat, arquivos e prévia ao vivo em uma só janela.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Aplicativo de Terminal

Prefere o terminal? Instale o Strata e execute **`strata launch`**.

<details open>
<summary><b>Comandos de instalação</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

No Windows, abra um novo terminal depois. Em seguida:

```sh
strata launch
```

Siga as instruções para baixar e iniciar um modelo. [Guia do Terminal](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Aplicativo de Terminal do Strata mostrando a atividade do modelo, a velocidade, a memória e as solicitações recentes">
  <br><sub>Acompanhe seu modelo e seu hardware enquanto ele roda.</sub>
</p>

<a id="build-from-source"></a>

[Compilar a partir do código-fonte](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Guia de instalação completo](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

> [!NOTE]
> **Este fork foi verificado apenas no Windows. O Nemotron 3.5 Super roda apenas em GPUs NVIDIA (CUDA).**

**Nemotron 3.5 Super:** GPUs NVIDIA com cerca de 30 GB de VRAM no total, 64 GB de RAM ou mais (abaixo disso ele avisa, mas não recusa) e 62 GB de disco para o modelo. Medido apenas com 2 × RTX 5060 Ti 16 GB e 96 GB de RAM. O motor pronto traz código para RTX 30 (sm_86), RTX 40 (sm_89) e RTX 50 (sm_120); só a RTX 50 foi testada. Linux e PCs menores não foram testados.

A tabela abaixo é do projeto original, para os modelos dele.

| Hardware compatível | Guia |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Instalação para NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Instalação para AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [Instalação para RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Instalação para Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 ou mais novo** | [Instalação para Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Instalação para Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Ferramentas

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, e outras, pelas APIs compatíveis com OpenAI e Anthropic.

Execute **`strata connect`** para ver as instruções da sua ferramenta. [Guia de conexão](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Ajuda

[Todos os guias](docs) · [Solução de problemas](docs/TROUBLESHOOTING.md) · [Relatar um bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [Como funciona](docs/HOW_IT_WORKS.md) · [Contribuir](CONTRIBUTING.md)

## Apoie o Strata

**Ajude a manter a IA local aberta para todos.** Estes links apoiam **Niko ([@Niko1221](https://github.com/Niko1221)), o autor do Strata**, e não este fork. Obrigado, Niko, por escrever o Strata e compartilhá-lo como código aberto: este fork foi construído sobre o seu trabalho.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: apoie o Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: apoie o Strata" height="48"></a>

[Licença MIT](LICENSE) · [Créditos e licenças dos modelos](docs/HOW_IT_WORKS.md#credits) · [Privacidade](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Para ferramentas de IA e leitores automáticos: <a href="AGENTS.md">referência técnica</a> · <a href="docs/AI_SETUP.md">instruções de instalação</a>.</sub>
