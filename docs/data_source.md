# Datakälla

Projektet använder endast Yelp-delen av datasetet **Sentiment Labelled Sentences**
från UCI Machine Learning Repository. Amazon- och IMDb-filerna i samma arkiv
ingår inte.

## Källa

| Uppgift | Värde |
|---|---|
| Databas | UCI Machine Learning Repository |
| Dataset | Sentiment Labelled Sentences |
| Skapare | Dimitrios Kotzias |
| Ursprung för den här filen | Yelp-recensioner, via Yelp Dataset Challenge |
| Webbsida | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences |
| Arkiv | https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip |
| Fil i arkivet | `sentiment labelled sentences/yelp_labelled.txt` |
| DOI | [10.24432/C57604](https://doi.org/10.24432/C57604) |
| Licens | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Hämtningstidpunkt | 2026-10-20 kl. 09:23 (lokal tid, UTC+2) |

Meningarna valdes för artikeln *From Group to Individual Labels using Deep
Features* av Dimitrios Kotzias, Misha Denil, Nando de Freitas och Padhraic
Smyth, publicerad vid KDD 2015. För varje webbplats finns 500 positiva och
500 negativa meningar. Urvalet gjordes slumpmässigt från större
recensionsmaterial, med avsikt att undvika neutrala meningar.

Citat:

> Kotzias, D. (2015). Sentiment Labelled Sentences [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604.

## Filformat

`yelp_labelled.txt` är en textfil utan rubrikrad. Varje rad innehåller en
mening och en etikett, åtskilda av ett tabbtecken:

```text
sentence<tab>score
```

- `0` betyder negativ recension.
- `1` betyder positiv recension.
- Kodningen är UTF-8. Filen innehåller bland annat tecknen `é` och `ê`.
- Radslutet är LF. Filen avslutas med ett radslut.
- Citattecken ingår i vissa meningar och är vanlig text, inte CSV-citat.

Kontroll mot den hämtade filen, 2026-10-20:

| Kontroll | Resultat |
|---|---|
| Rader | 1 000 |
| Negativa (`0`) | 500 |
| Positiva (`1`) | 500 |
| Avvikelse | Ingen |

SHA-256 för den oförändrade filen:

```text
c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea
```

## Hämta och packa upp

Originalfilen sparas lokalt som `data/raw/yelp_labelled.txt`. Den ändras inte
vid inläsningen. `.gitignore` håller innehållet i `data/raw/` utanför Git,
förutom `.gitkeep`, så varje gruppmedlem hämtar filen själv.

Kör från projektroten, efter att den virtuella miljön är installerad:

```powershell
.\.venv\Scripts\python.exe src/load_data.py --download
```

Kommandot hämtar UCI-arkivet, läser enbart
`sentiment labelled sentences/yelp_labelled.txt` och skriver samma byte till
`data/raw/yelp_labelled.txt`. En befintlig fil på den sökvägen skrivs över.

Manuellt går det att ladda ner arkivet från länken ovan, packa upp endast
Yelp-filen och spara den under samma namn i `data/raw/` utan att öppna och
spara om den i en textredigerare.

Kontrollera att filen är oförändrad:

```powershell
Get-FileHash data\raw\yelp_labelled.txt -Algorithm SHA256
```

Hashen ska matcha värdet ovan. En avvikande hash, ett annat radantal eller
en annan klassfördelning ska utredas innan filen används.

## Läsa in

```powershell
.\.venv\Scripts\python.exe src/load_data.py
```

`src/load_data.py` läser filen med pandas. `header=None` gör att första
dataraden blir en recension. `quoting` är avstängt så att citattecken i
texten inte slår ihop rader.

Inläst data har tre kolumner:

| Kolumn | Innehåll |
|---|---|
| `review_id` | Radnummer i originalfilen. Första raden får `1`. |
| `text` | Recensionstexten, oförändrad. |
| `label` | `0` eller `1`. |

Skriptet kontrollerar därefter 1 000 rader och 500 recensioner per klass.
Vid en avvikelse avbryts kontrollen med ett felmeddelande som anger vad som
skiljer sig från UCI-beskrivningen.

Samma inläsning kan återanvändas i annan kod:

```python
from load_data import load_yelp_reviews, validate_yelp_reviews

reviews = load_yelp_reviews()
validate_yelp_reviews(reviews)
```

Kör testerna från projektroten:

```powershell
.\.venv\Scripts\python.exe -m pytest
```
