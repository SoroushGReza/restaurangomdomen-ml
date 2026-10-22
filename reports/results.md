# Slututvärdering på testdata

## Metod

Baseline och textmodellen är de sparade modellerna från uppgift 05 och 06.
Ingen modell anpassas här, och testdata används inte för modellval.
Båda modellerna får exakt samma testrader. Prediktionerna matchas mot facit via `review_id`.
Confusion matrix har klassordningen [0, 1], med verklig klass på raderna och förutsagd klass i kolumnerna.
Valideringsresultaten ligger kvar i `reports/model_selection.md` och blandas inte in i `reports/metrics_test.csv`.

## Personuppgifter

Datakvalitetskontrollen identifierade 4 möjliga namnträffar.
Skolans krav gäller behandling av personuppgifter i projektet, inte enbart publicering av exempel.
Gruppen behöver stämma av med läraren hur träffarna ska hanteras innan fortsatt behandling.
Den här utvärderingen gör ingen ny bearbetning av råfilen och citerar inga recensionstexter.

## Testunderlag

- Rader: 200.
- Negativa (0): 100.
- Positiva (1): 100.
- Träning, validering och test har giltiga etiketter, inga tomma texter och inga gemensamma ID:n eller dubblettgrupper.

## Modellinställningar

- Baseline: alltid klass 1, beräknad tidigare från träningsmaterialet.
- Textmodell: `tfidf_unigram_bigram`.
- TF-IDF `ngram_range=(1, 2)`, `stop_words=None`.
- LogisticRegression `C=1.0`, `max_iter=1000`, `random_state=42`.
- Scikit-learn 1.9.1.

## Jämförelse

Högre värde är bättre för alla tre mått. Tabellen redovisar det faktiska testresultatet,
även om textmodellen inte är bättre än baseline på ett mått.

| Mått | majority_baseline | tfidf_unigram_bigram | Bäst |
| --- | --- | --- | --- |
| Macro-F1 | 0.333333 | 0.744841 | tfidf_unigram_bigram |
| Negativ recall | 0.000000 | 0.720000 | tfidf_unigram_bigram |
| Accuracy | 0.500000 | 0.745000 | tfidf_unigram_bigram |

| Mått | majority_baseline | tfidf_unigram_bigram |
| --- | --- | --- |
| Precision klass 0 | 0.000000 | 0.757895 |
| Recall klass 0 | 0.000000 | 0.720000 |
| F1 klass 0 | 0.000000 | 0.738462 |
| Precision klass 1 | 0.500000 | 0.733333 |
| Recall klass 1 | 1.000000 | 0.770000 |
| F1 klass 1 | 0.666667 | 0.751220 |

## Confusion matrix

- majority_baseline: verklig 0 förutsades som [0: 0, 1: 100]; verklig 1 förutsades som [0: 0, 1: 100]. Modellen förutsäger aldrig klass 0. Precision för den klassen sätts till 0 eftersom det inte finns några sådana prediktioner.
- tfidf_unigram_bigram: verklig 0 förutsades som [0: 72, 1: 28]; verklig 1 förutsades som [0: 23, 1: 77]. Modellen förutsäger båda klasserna.

Figurerna finns i `reports/figures/`.

## Begränsningar

Underlaget är ett litet engelskspråkigt Yelp-urval utan neutral klass.
Resultatet visar inte hur modellerna fungerar på svenska eller på alla restaurangers omdömen.
Valideringsdata användes för modellvalet och ingår inte i den här jämförelsen.
Baseline kan få en accuracy nära klassbalansen samtidigt som negativ recall blir 0,
eftersom den alltid väljer samma klass.

## Regel efter sluttestet

Modellinställningarna ändras inte efter granskning av testresultatet.
Felanalysen finns i `reports/error_analysis.md` och riskerna i `docs/risks.md`.
Svenska texter och en eventuell neutral klass är fortfarande framtida arbete.
