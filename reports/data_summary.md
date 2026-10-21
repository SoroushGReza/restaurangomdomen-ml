# Datauppdelning och inledande analys

Baseline och den jämförande modellen ska använda samma delmängder.
Uppdelningen görs innan någon textvektor eller modell anpassas.

## Metod

Underlaget är `data/processed/reviews.csv` från datakvalitetskontrollen.
En dubblettgrupp identifieras med `duplicate_group_id`.
Alla recensioner i samma grupp placeras i samma delmängd.

- Träning: 60 % av dubblettgrupperna.
- Validering: 20 % av dubblettgrupperna.
- Test: 20 % av dubblettgrupperna.
- `random_state=42`.
- Stratifiering görs på gruppens etikett, så att klassfördelningen behålls så långt det går.
- Om en grupp har flera etiketter används majoriteten. Vid lika antal används etiketten på det lägsta `review_id`.

Varje dubblettgrupp innehåller en rad. Radandelarna följer därför gruppdelningen direkt.
Antal grupper med mer än en etikett: 0.

## Klassfördelning

| Delmängd | Rader | Andel | Grupper | Negativa (0) | Positiva (1) | Andel positiva |
| --- | --- | --- | --- | --- | --- | --- |
| Träning | 597 | 59.9 % | 597 | 298 | 299 | 50.1 % |
| Validering | 199 | 20.0 % | 199 | 99 | 100 | 50.3 % |
| Test | 200 | 20.1 % | 200 | 100 | 100 | 50.0 % |

Totalt antal rader: 996.
Totalt antal dubblettgrupper: 996.

Inget `review_id` och ingen `duplicate_group_id` förekommer i mer än en delmängd.

## Textlängd i träningsdata

Längden räknas i tecken i originaltexten. Ingen tokenisering eller annan
modellförbehandling ingår. Validerings- och testtexter citeras inte,
eftersom de inte ska styra modellutvecklingen.

| Urval | Rader | Min | Q1 | Median | Medel | Q3 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Träning, alla | 597 | 11 | 33 | 50 | 58.0 | 80 | 149 |
| Träning, negativa | 298 | 13 | 32 | 49.5 | 59.0 | 82 | 149 |
| Träning, positiva | 299 | 11 | 33 | 50 | 56.9 | 77 | 148 |

Träningstexterna är korta recensionsmeningar. Medianlängden är 50 tecken och hälften av texterna ligger mellan 33 och 80 tecken.

## Representativa träningstexter

Exemplen kommer bara från träningsdata: en kort, en typisk och en lång text per klass.
Recensioner som datakvalitetskontrollen markerade som möjliga personnamn skrivs inte ut.
Skolans krav gäller behandling av personuppgifter i projektet, inte enbart publicering av exempel.
Gruppen behöver stämma av med läraren hur träffarna ska hanteras innan fortsatt behandling.

| Typ | Klass | review_id | Tecken | Text |
| --- | --- | --- | --- | --- |
| kort | negativ | 671 | 13 | dont go here. |
| median | negativ | 476 | 49 | Third, the cheese on my friend's burger was cold. |
| lång | negativ | 986 | 149 | The problem I have is that they charge $11.99 for a sandwich that is no bigger than a Subway sub (which offers better and more amount of vegetables). |
| kort | positiv | 649 | 11 | Great food. |
| median | positiv | 191 | 49 | you can watch them preparing the delicious food!) |
| lång | positiv | 550 | 148 | My boyfriend and I came here for the first time on a recent trip to Vegas and could not have been more pleased with the quality of food and service. |

## Användning

- Träningsdata används för att anpassa textomvandling och modeller.
- Valideringsdata används för modellval.
- Testdata används först när modellvalet är låst.
- Testresultat får inte styra modellutvecklingen.

Filerna sparas lokalt som `data/splits/train.csv`, `data/splits/validation.csv`
och `data/splits/test.csv`. Mappen `data/splits/` är gitignorerad, förutom `.gitkeep`.

Kör från projektroten:

```powershell
.\.venv\Scripts\python.exe src/split_data.py
```
