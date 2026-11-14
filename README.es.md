<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">IA potente en tu propio ordenador.</h3>
<p align="center">Chatea, programa y trabaja con imágenes, en local.</p>

<p align="center">
  <a href="#modelos">Modelos</a> · <a href="#instalación"><b>Instalación</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#herramientas">Herramientas</a> ·
  <a href="#ayuda">Ayuda</a> · <a href="#apoya-strata">Apoyar</a>
</p>

<p align="center">
  <a href="https://github.com/shiosalt/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Descargar Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licencia MIT"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Patrocina Strata en GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Apoya Strata en Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <b>Español</b> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

> [!IMPORTANT]
> **Este es un fork no oficial** ([shiosalt/Strata](https://github.com/shiosalt/Strata)) de [Niko1221/Strata](https://github.com/Niko1221/Strata). Añade **NVIDIA-Nemotron-3.5-Super-VL-120B-A12B** con un motor propio. Los desarrolladores de Strata no escribieron ni revisaron esta parte: informa de sus problemas [aquí](https://github.com/shiosalt/Strata/issues), no al proyecto original. El resto de este README es del proyecto original, igual que los instaladores y los enlaces de versiones de abajo (no incluyen el motor de Nemotron). Mira [Modelos](#nemotron-models).

**Gratis, de código abierto y tuyo para ejecutar.** Usa Strata en tu escritorio o con tus herramientas de programación favoritas. No necesitas cuenta.

## Modelos

<a id="nemotron-models"></a>

### <img src="docs\media\models\nvidia-color.svg" width="32" alt="nvidia" align="middle"> &nbsp; Nemotron 3.5 Super (este fork)

[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF): 120 000 millones de parámetros, 12 000 millones activos por token, lee imágenes. Este fork ejecuta **UD-IQ3_XXS** de Unsloth (59,8 GB más un lector de imágenes de 1,7 GB) con un motor propio, en la Terminal. Funcionan texto, razonamiento, llamadas a herramientas, imágenes, borradores MTP y contexto de 128K / 256K. Responde a una petición cada vez.

Medido en un solo PC (2 × RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB de RAM, Windows 11, contexto de 128K):

| | Velocidad |
| --- | --- |
| Generación, petición corta (temperatura 1,0, MTP 6) | 24-28 tokens/s |
| Generación, a 72K tokens de una conversación | 12.7-15.4 tokens/s |
| Lectura de un prompt de 36.529 tokens | 276 tokens/s (132 s) |

Cómo instalarlo: [Instalación](#nemotron-install). Los pesos son de NVIDIA, bajo la licencia de su ficha de modelo; este repositorio no incluye ninguno. Detalles y todas las mediciones (en japonés): [docs/NEMOTRON35.md](docs/NEMOTRON35.md).

**Con las funciones originales de Strata también se pueden usar estos modelos:**

Recomendamos **primero ISTA-DASLab**, después **Unsloth** y otros proveedores, siempre que haya un modelo compatible que quepa en tu memoria.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Proveedor | Opciones |
| --- | --- |
| **ISTA-DASLab · recomendado** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) para programar. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Usa menos memoria, pero es más lento. |
| **Otras opciones** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compara tamaños, necesidades de memoria y compatibilidad](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Son los paquetes documentados actualmente para este modelo.

[Guía del modelo y compatibilidad](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

En la **aplicación de escritorio** y en el **Terminal**. Lo que no cabe en la memoria se lee desde tu SSD, así que hace falta mucho espacio libre en disco.

| Conjunto (ISTA-DAS) | Descarga | Espacio libre necesario |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | unos 245 GB |
| 2.00 bpw | 257 GB | unos 265 GB |
| 2.25 bpw | 274 GB | unos 280 GB |

Funciona en NVIDIA RTX 30, 40 y 50 (Windows y Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo (Windows) y Macs con Apple Silicon de 64 GB o más ([Mac](docs/DEEPSEEK_MAC.md)). La instalación descarga los archivos y el motor de DeepSeek, y después inicia el modelo.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Medido con el conjunto de 2.00 bpw en Strata 0.2.5. «Salida» es la velocidad de la respuesta; «Prompt» es la rapidez con la que Strata lee tu texto.

| Hardware | Salida | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB de RAM | 4 a 6 tokens/s | unos 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB de RAM | 9 a 11 tokens/s | unos 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB de RAM | 21 a 25 tokens/s | 320 a 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 a 77 tokens/s | 400 a 450 tokens/s (8K) |

[Memoria, disco y detalles](docs/DEEPSEEK41.md) · [Notas de la versión](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

¿No sabes qué cabe? [Elige un modelo para tu ordenador](docs/MODELS.md#pick-by-memory).

## Instalación

<a id="nemotron-install"></a>

> [!NOTE]
> **Este fork solo se ha comprobado en Windows. Nemotron 3.5 Super solo funciona en GPU NVIDIA (CUDA).** La aplicación de escritorio y los instaladores originales de abajo no lo ejecutan: usa la Terminal de este repositorio.

**Nemotron 3.5 Super con el motor ya compilado (necesita Python 3.10+, Git y un controlador NVIDIA 580 o más reciente):**

1. Descarga este repositorio: `git clone https://github.com/shiosalt/Strata.git` y después `cd Strata`.
2. Descarga `strata-nemotron-0.1.0-windows-x64-cuda13.zip` desde [Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) y descomprímelo en la carpeta del repositorio (crea `engine-nemotron\`).
3. Instala los paquetes de Python:

   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8
   ```

4. Descarga el modelo (o indica tus archivos con `--model-path`) e inícialo:

   ```bat
   .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
   ```

   `.venv\Scripts\python.exe strata.py launch --stop` lo detiene. El comando `strata` instalado es del proyecto original y no conoce este modelo.

Para compilar el motor tú mismo (Visual Studio 2022 Build Tools con «Desarrollo para el escritorio con C++» y CUDA Toolkit 12.8 o más reciente; 30 minutos o más), ejecuta `powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision` en lugar de los pasos 2 y 3.

El resto de esta sección es del proyecto original, para sus modelos.

<a id="option-a-the-strata-app"></a>

### Aplicación de escritorio

**[Descargar Strata para Windows, Linux o Mac →](https://github.com/shiosalt/Strata/releases/latest)**

Instala la aplicación, elige un modelo y empieza a chatear. Hay un instalador por sistema. En el primer arranque, la aplicación descarga el motor para tu tarjeta gráfica. [Ayuda con el instalador](docs/INSTALL.md#the-app).

<sub>Archivos: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Aplicación de escritorio de Strata con un chat y una vista previa del código en vivo, una junto a otra">
  <br><sub>Chat, archivos y vista previa en vivo en una sola ventana.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Aplicación de Terminal

¿Prefieres la terminal? Instala Strata y ejecuta **`strata launch`**.

<details open>
<summary><b>Comandos de instalación</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/shiosalt/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/shiosalt/Strata/releases/latest/download/install.ps1 | iex
```

En Windows, abre una terminal nueva después. Luego:

```sh
strata launch
```

Sigue las indicaciones para descargar e iniciar un modelo. [Guía del Terminal](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Aplicación de Terminal de Strata con la actividad del modelo, la velocidad, la memoria y las peticiones recientes">
  <br><sub>Observa tu modelo y tu hardware mientras se ejecuta.</sub>
</p>

<a id="build-from-source"></a>

[Compilar desde el código fuente](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Guía de instalación completa](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

> [!NOTE]
> **Este fork solo se ha comprobado en Windows. Nemotron 3.5 Super solo funciona en GPU NVIDIA (CUDA).**

**Nemotron 3.5 Super:** GPU NVIDIA con unos 30 GB de VRAM en total, 64 GB de RAM o más (por debajo avisa, pero no se niega) y 62 GB de disco para el modelo. Medido solo con 2 × RTX 5060 Ti 16 GB y 96 GB de RAM. El motor ya compilado incluye código para RTX 30 (sm_86), RTX 40 (sm_89) y RTX 50 (sm_120); solo se probó con RTX 50. No se probó en Linux ni en PC más pequeños.

La tabla de abajo es del proyecto original, para sus modelos.

| Hardware compatible | Guía |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Instalación en NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Instalación en AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [Instalación en RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Instalación en Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 o posterior** | [Instalación en Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Instalación en Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Herramientas

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, y más, mediante las API compatibles con OpenAI y Anthropic.

Ejecuta **`strata connect`** para ver las instrucciones de tu herramienta. [Guía de conexión](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Ayuda

[Todas las guías](docs) · [Solución de problemas](docs/TROUBLESHOOTING.md) · [Informar de un error](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [Cómo funciona](docs/HOW_IT_WORKS.md) · [Contribuir](CONTRIBUTING.md)

## Apoya Strata

**Ayuda a mantener la IA local abierta a todos.** Estos enlaces apoyan a **Niko ([@Niko1221](https://github.com/Niko1221)), el autor de Strata**, no a este fork. Gracias, Niko, por escribir Strata y compartirlo como código abierto: este fork se basa en tu trabajo.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: patrocina Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: apoya Strata" height="48"></a>

[Licencia MIT](LICENSE) · [Créditos y licencias de los modelos](docs/HOW_IT_WORKS.md#credits) · [Privacidad](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Para herramientas de IA y lectores automáticos: <a href="AGENTS.md">referencia técnica</a> · <a href="docs/AI_SETUP.md">instrucciones de instalación</a>.</sub>
