<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Uruchom potężną sztuczną inteligencję na własnym komputerze.</h3>
<p align="center">Rozmawiaj, programuj i pracuj z obrazami — lokalnie.</p>

<p align="center">
  <a href="#modele">Modele</a> · <a href="#instalacja"><b>Instalacja</b></a> ·
  <a href="#sprzęt">Sprzęt</a> · <a href="#narzędzia">Narzędzia</a> ·
  <a href="#pomoc">Pomoc</a> · <a href="#wesprzyj-stratę">Wsparcie</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Pobierz Stratę"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licencja MIT"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Wesprzyj Stratę na GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Wesprzyj Stratę na Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <b>Polski</b></sub></p>

> [!IMPORTANT]
> **To jest nieoficjalny fork** ([shiosalt/Strata](https://github.com/shiosalt/Strata)) projektu [Niko1221/Strata](https://github.com/Niko1221/Strata). Dodaje **NVIDIA-Nemotron-3.5-Super-VL-120B-A12B** z własnym silnikiem. Twórcy Straty nie napisali ani nie sprawdzili tej części: problemy z nią zgłaszaj [tutaj](https://github.com/shiosalt/Strata/issues), a nie w projekcie oryginalnym. Reszta tego README pochodzi z oryginału, tak jak instalatory i linki do wydań poniżej (nie zawierają silnika Nemotron). Zobacz [Modele](#nemotron-models).

**Bezpłatna, z otwartym kodem źródłowym i do uruchomienia u siebie.** Używaj Straty na swoim komputerze albo razem z ulubionymi narzędziami do programowania. Konto nie jest potrzebne.

## Modele

<a id="nemotron-models"></a>

### Nemotron 3.5 Super (ten fork)

[NVIDIA-Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF): 120 mld parametrów, 12 mld aktywnych na token, czyta obrazy. Ten fork uruchamia **UD-IQ3_XXS** od Unsloth (59,8 GB oraz czytnik obrazów 1,7 GB) na własnym silniku, w Terminalu. Działają tekst, myślenie, wywołania narzędzi, obrazy, szkice MTP i kontekst 128K / 256K. Odpowiada na jedno żądanie naraz.

Zmierzone tylko na jednym komputerze (2 × RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB RAM, Windows 11, kontekst 128K):

| | Szybkość |
| --- | --- |
| Generowanie, krótkie żądanie (temperatura 1,0, MTP 6) | 24-28 tokens/s |
| Generowanie, 72K tokenów w głąb rozmowy | 12.7-15.4 tokens/s |
| Wczytanie promptu o długości 36 529 tokenów | 276 tokens/s (132 s) |

Instalacja: [Instalacja](#nemotron-install). Wagi należą do NVIDII i podlegają licencji z karty modelu; to repozytorium ich nie zawiera. Szczegóły i wszystkie pomiary (po japońsku): [docs/NEMOTRON35.md](docs/NEMOTRON35.md).

**Dzięki oryginalnym funkcjom Straty można też używać tych modeli:**

Najpierw polecamy **ISTA-DASLab**, potem **Unsloth** i innych dostawców — o ile zgodny model mieści się w pamięci.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Dostawca | Do wyboru |
| --- | --- |
| **ISTA-DASLab · polecany** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) do programowania. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Zużywa mniej pamięci, ale działa wolniej. |
| **Inne opcje** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Porównaj rozmiary, wymagania pamięciowe i zgodność](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. To obecnie udokumentowane pakiety dla tego modelu.

[Przewodnik po modelu i zgodność](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

W **aplikacji na komputer** i w **Terminalu**. To, co nie mieści się w pamięci, jest czytane z dysku SSD, więc potrzebujesz
dużo wolnego miejsca na dysku.

| Zestaw (ISTA-DAS) | Do pobrania | Potrzebne wolne miejsce |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | około 245 GB |
| 2.00 bpw | 257 GB | około 265 GB |
| 2.25 bpw | 274 GB | około 280 GB |

Działa na kartach NVIDIA RTX serii 30, 40 i 50 (Windows i Linux), AMD Radeon AI PRO R9700 (Linux) oraz AMD Ryzen AI Max — Strix Halo
(Windows) oraz na komputerach Mac z Apple Silicon i co najmniej 64 GB pamięci ([Mac](docs/DEEPSEEK_MAC.md)). Instalacja pobiera pliki i silnik DeepSeek, a potem uruchamia model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Zmierzone na zestawie 2.00 bpw w Strata 0.2.5 (Strix Halo: 0.2.7). „Wyjście” to szybkość odpowiedzi, „prompt” — jak szybko Strata czyta twój tekst.

| Sprzęt | Wyjście | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4–6 tokenów/s | około 110 tokenów/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9–11 tokenów/s | około 155 tokenów/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21–25 tokenów/s | 320–360 tokenów/s (8K) |
| 4x Radeon AI PRO R9700 | 74–77 tokenów/s | 400–450 tokenów/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB pamięci współdzielonej | 19–21 tokenów/s (kod), 11 (czat) | 87 tokenów/s (4K), 125 tokenów/s (8K) |

[Pamięć, dysk i szczegóły](docs/DEEPSEEK41.md) · [Informacje o wydaniach](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Nie wiesz, co się zmieści? [Wybierz model do swojego komputera](docs/MODELS.md#pick-by-memory).

## Instalacja

<a id="nemotron-install"></a>

> [!NOTE]
> **Ten fork sprawdzono tylko w systemie Windows. Nemotron 3.5 Super działa tylko na kartach NVIDIA (CUDA).** Aplikacja na komputer i oryginalne instalatory poniżej go nie uruchomią: użyj Terminala z tego repozytorium.

**Nemotron 3.5 Super z gotowym silnikiem (potrzebne: Python 3.10+, Git i sterownik NVIDIA 580 lub nowszy):**

1. Pobierz to repozytorium: `git clone https://github.com/shiosalt/Strata.git`, potem `cd Strata`.
2. Pobierz `strata-nemotron-0.1.0-windows-x64-cuda13.zip` z [Releases](https://github.com/shiosalt/Strata/releases/tag/nemotron-v0.1.0) i rozpakuj go do folderu repozytorium (powstanie `engine-nemotron\`).
3. Zainstaluj pakiety Pythona:

   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt textual==8.2.8
   ```

4. Pobierz model (albo wskaż swoje pliki przez `--model-path`) i uruchom go:

   ```bat
   .venv\Scripts\python.exe strata.py launch --family nemotron35 --model UD-IQ3_XXS --yes --no-tui
   ```

   `.venv\Scripts\python.exe strata.py launch --stop` go zatrzymuje. Zainstalowane polecenie `strata` pochodzi z oryginału i nie zna tego modelu.

Aby zamiast tego samodzielnie zbudować silnik (Visual Studio 2022 Build Tools z „Programowaniem aplikacji klasycznych w C++” i CUDA Toolkit 12.8 lub nowszy; 30 minut lub dłużej), uruchom `powershell -ExecutionPolicy Bypass -File tools\nemotron\build.ps1 -Vision` zamiast kroków 2 i 3.

Dalsza część tej sekcji pochodzi z oryginału i dotyczy jego modeli.

<a id="option-a-the-strata-app"></a>

### Aplikacja na komputer

**[Pobierz Stratę dla Windows, Linuxa lub Maca →](https://github.com/Niko1221/Strata/releases/latest)**

Zainstaluj aplikację, wybierz model i zacznij rozmawiać. Dla każdego systemu jest jeden instalator. Przy pierwszym
uruchomieniu aplikacja pobiera silnik dla twojej karty graficznej. [Pomoc z instalatorem](docs/INSTALL.md#the-app).

<sub>Pliki: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Aplikacja Strata na komputer: czat i podgląd kodu na żywo obok siebie">
  <br><sub>Czat, pliki i podgląd na żywo w jednym oknie.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Aplikacja Terminal

Wolisz terminal? Zainstaluj Stratę, a potem uruchom **`strata launch`**.

<details open>
<summary><b>Polecenia instalacji</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

W Windows otwórz potem nowe okno terminala. Następnie:

```sh
strata launch
```

Postępuj zgodnie z podpowiedziami, aby pobrać i uruchomić model. [Przewodnik po Terminalu](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Aplikacja Terminal Straty pokazująca aktywność modelu, szybkość, pamięć i ostatnie zapytania">
  <br><sub>Obserwuj model i sprzęt w trakcie pracy.</sub>
</p>

<a id="build-from-source"></a>

[Budowanie ze źródeł](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Pełny przewodnik instalacji](docs/INSTALL.md)

<a id="what-you-need"></a>

## Sprzęt

> [!NOTE]
> **Ten fork sprawdzono tylko w systemie Windows. Nemotron 3.5 Super działa tylko na kartach NVIDIA (CUDA).**

**Nemotron 3.5 Super:** karty NVIDIA z łącznie około 30 GB VRAM, 64 GB RAM lub więcej (poniżej tego ostrzega, ale nie odmawia) i 62 GB miejsca na dysku na model. Zmierzone tylko na 2 × RTX 5060 Ti 16 GB z 96 GB RAM. Gotowy silnik zawiera kod dla kart RTX 30 (sm_86), RTX 40 (sm_89) i RTX 50 (sm_120); wypróbowano tylko RTX 50. Linuksa i mniejszych komputerów nie wypróbowano.

Tabela poniżej pochodzi z oryginału i dotyczy jego modeli.

| Obsługiwany sprzęt | Przewodnik |
| --- | --- |
| **NVIDIA GeForce RTX serii 20 / 30 / 40 / 50** | [Konfiguracja NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Konfiguracja AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · eksperymentalnie** | [Konfiguracja RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Konfiguracja Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 lub nowszy** | [Konfiguracja Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Konfiguracja Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Narzędzia

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw** i inne — przez API zgodne z OpenAI i Anthropic.

Uruchom **`strata connect`**, aby zobaczyć instrukcję konfiguracji dla swojego narzędzia. [Przewodnik po połączeniach](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Pomoc

[Wszystkie przewodniki](docs) · [Rozwiązywanie problemów](docs/TROUBLESHOOTING.md) · [Zgłoś błąd](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Testy wydajności](docs/COMMUNITY_BENCHMARKS.md) · [Jak to działa](docs/HOW_IT_WORKS.md) · [Współtworzenie](CONTRIBUTING.md)

## Wesprzyj Stratę

**Pomóż zachować lokalną sztuczną inteligencję otwartą dla wszystkich.** Te linki wspierają **Niko ([@Niko1221](https://github.com/Niko1221)), autora Straty**, a nie ten fork. Dziękuję, Niko, za napisanie Straty i udostępnienie jej jako open source: ten fork powstał na bazie Twojej pracy.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: wesprzyj Stratę" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: wesprzyj Stratę" height="48"></a>

[Licencja MIT](LICENSE) · [Podziękowania i licencje modeli](docs/HOW_IT_WORKS.md#credits) · [Prywatność](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Dla narzędzi AI i automatycznych czytników: <a href="AGENTS.md">dokumentacja techniczna</a> · <a href="docs/AI_SETUP.md">instrukcja instalacji</a>.</sub>
