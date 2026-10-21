## Installation och utvecklingsmiljö

Instruktionerna gäller Windows med PowerShell i VS Code.

### Förutsättningar

- Git.
- Python enligt versionen i `python-version.txt`.
- VS Code med tillägget Python från Microsoft.
- Åtkomst till projektets GitHub-repository.

Paketversionerna finns i `requirements.txt`. Varje gruppmedlem skapar
en egen lokal virtuell miljö i `.venv`.

### 1. Hämta projektet

Vid första installationen:

```powershell
git clone https://github.com/SoroushGReza/restaurangomdomen-ml.git
cd restaurangomdomen-ml
```

Öppna projektmappen i VS Code.

Om projektet redan är klonat, öppna den befintliga projektmappen
och hämta senaste versionen:

```powershell
git pull --ff-only origin main
```

### 2. Skapa en virtuell miljö

Kör från projektroten. Kontrollera Pythonversionen mot
`python-version.txt` innan miljön skapas.

```powershell
python --version
python -m venv .venv
```

### 3. Installera projektets beroenden

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kommandona använder Python från projektets virtuella miljö direkt.

### 4. Kontrollera installationen

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Förväntat resultat:

```text
No broken requirements found.
```

Kontrollera också att projektets paket kan importeras:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, sklearn, matplotlib, joblib, pytest; print('Environment ready')"
```

Förväntat resultat:

```text
Environment ready
```

### 5. Välj miljön i VS Code

1. Tryck `Ctrl + Shift + P`.
2. Välj `Python: Select Interpreter`.
3. Välj `.venv\Scripts\python.exe`.

### Fortsatt arbete

Den virtuella miljön skapas en gång per dator.

Hämta senaste ändringarna inför ett nytt arbetspass:

```powershell
git pull --ff-only origin main
```

Om `requirements.txt` har ändrats, uppdatera miljön:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ta en ledig issue, tilldela den till dig själv och flytta den till
`In progress`. Dokumentera framsteg och eventuella hinder i issuen.

### Projektstruktur

| Mapp | Innehåll |
|---|---|
| `src/` | Kod för datahantering, modeller, utvärdering och demo |
| `data/raw/` | Lokala originaldata |
| `data/processed/` | Lokala bearbetade data |
| `data/splits/` | Lokala tränings-, validerings- och testdata |
| `models/` | Lokalt sparade modeller |
| `reports/` | Analyser och dokumenterade resultat |
| `reports/figures/` | Diagram och resultatfigurer |
| `docs/` | Datakälla, beslut, risker och AI-användning |
| `tests/` | Automatiska tester |

`.gitkeep` används för att tomma projektmappar ska följa med i Git.
`.gitignore` håller virtuella miljöer, lokala dataset, genererade
modeller och hemliga miljövariabler utanför versionshanteringen.

### Projektets körning

### Läsa in Yelp-data

Hämtning, uppackning och kontroll av datakällan beskrivs i
`docs/data_source.md`. Kör från projektroten:

```powershell
.\.venv\Scripts\python.exe src/load_data.py --download
.\.venv\Scripts\python.exe src/load_data.py
.\.venv\Scripts\python.exe -m pytest
```

`--download` hämtar UCI-arkivet och sparar `data/raw/yelp_labelled.txt`
lokalt och oförändrad. Nästa kommando läser in filen och kontrollerar
1 000 rader och 500 recensioner per klass.

### Kontrollera datakvalitet

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py
```

Kommandot läser den lokala Yelp-filen, tar bort exakta dubbletter och
skriver `data/processed/reviews.csv` samt `reports/data_quality.md`.
Originaltexten behålls. Nära dubbletter får ett gemensamt grupp-ID.

### Dela data

```powershell
.\.venv\Scripts\python.exe src/split_data.py
```

Kommandot läser `data/processed/reviews.csv` och skriver
`data/splits/train.csv`, `data/splits/validation.csv` och
`data/splits/test.csv`. Uppdelningen använder `random_state=42`,
stratifieras på klass och håller samma `duplicate_group_id` i en delmängd.
Analysen skrivs till `reports/data_summary.md`.

Kommandon för modellträning, utvärdering och demonstration läggs till
när respektive del har implementerats.


### Majoritetsbaseline (uppgift 05)

`src/train_baseline.py` väljer den vanligaste klassen enbart från
`data/splits/train.csv`. Vid lika antal väljs klass `0`.
Recensionernas text används inte. Valideringsetiketterna påverkar inte
vilken klass som väljs.

Körningen förutsätter att den granskade datauppdelningen är klar och att
hanteringen av eventuella personuppgifter är avstämd enligt skolans krav.
Kör därefter från projektroten:

```powershell
.\.venv\Scripts\python.exe src/train_baseline.py
```

Kommandot skapar:

- `models/baseline.json`: vald klass och träningsmaterialets klassantal.
- `data/processed/baseline_validation_predictions.csv`: en prediktion per
  valideringsrad, med kolumnerna `review_id`, `y_true`, `y_pred`, `model` och `split`.
- `reports/baseline.md`: metod, klassantal och information om utdata.

Modellfilen och prediktionerna sparas lokalt och omfattas av befintlig
`.gitignore`. Rapporten versionshanteras efter körningen på projektets data.

Baseline ger en jämförelsenivå för att undersöka om textanalys förbättrar
klassificeringen. Mätvärden beräknas i den gemensamma utvärderingen.
Testdata läses inte av träningskommandot. Vid sluttestet kan
`load_baseline` och `predict_baseline` återanvända den sparade klassen
utan att anpassa modellen igen.

Testa baseline separat:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_train_baseline.py -q
```

Testerna använder syntetiska testfall och kräver inga Yelp-filer.
De kontrollerar bland annat majoritetsval, lika klassantal, prediktionsformat,
överlapp mellan delmängder och återanvändning av sparad baseline.
Syntetiska testfall används endast för kodkontroll; projektets resultat
ska baseras på Yelp-datan.

### Användning av generativ AI

Generativ AI har använts som stöd för planering, kodgranskning och dokumentation. 
