# Majoritetsbaseline

## Metod

Baseline väljer den vanligaste klassen enbart från träningsmaterialets etiketter.
Vid lika klassantal väljs klass 0. Samma klass förutsägs för varje valideringsrad.
Recensionernas text och innehåll används inte.

Baseline visar vad som kan uppnås utan textanalys och ger en enkel jämförelsenivå
för TF-IDF och logistisk regression. Majoritetsklassens dominans kan samtidigt
ge missvisande accuracy; jämförelsen ska därför även använda macro-F1 och
recall för negativa recensioner.

## Träningsmaterial

Källa: data/splits/train.csv

| Klass | Antal |
| --- | --- |
| Negativ (0) | 298 |
| Positiv (1) | 299 |
| Totalt | 597 |

Vald baselineklass: 1 (positiv).

## Valideringsprediktioner

- Källa: data/splits/validation.csv
- Antal prediktioner: 199
- Fil: data/processed/baseline_validation_predictions.csv
- Kolumner: review_id, y_true, y_pred, model, split.

Utvärderingsmåtten beräknas i projektets gemensamma utvärdering.
Det här steget läser inte testdata och redovisar inga testresultat.

## Återanvändning vid sluttest

Modellen sparas lokalt i models/baseline.json.
Funktionen load_baseline laddar den sparade klassen. predict_baseline kan
därefter användas med split="test" utan att majoritetsklassen beräknas om.
