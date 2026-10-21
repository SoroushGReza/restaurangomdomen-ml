# Modellval: TF-IDF och logistisk regression

## Metod

En Pipeline anpassar TF-IDF och LogisticRegression enbart på train.csv.
Endast text används som indata. ID och etikett används inte som egenskaper.
Två alternativ jämförs: enskilda ord (1, 1) samt enskilda ord och ordpar (1, 2).
Gemensamma inställningar: C=1.0, max_iter=1000, random_state=42,
stop_words=None. Övriga inställningar är scikit-learns standardvärden.
Scikit-learn-version: 1.9.1.

## Underlag

- Träningsrader: 597.
- Valideringsrader: 199.
- SHA256 train.csv: `0a3e1db4cbcea2ca015144468c04ff129276dfc9f33a46ec7400c036a8505e66`.
- SHA256 validation.csv: `dab39aad6e49ba8619aaad5a1ebb508c56c002f9818ecc177e1cad031d652c71`.
- Baseline beräknas på exakt samma träningsetiketter som textmodellerna.
- Alla alternativ utvärderas med samma evaluate_predictions och samma valideringsrader.

## Valideringsresultat

| Modell | ngram_range | Macro-F1 | Negativ recall |
| --- | --- | --- | --- |
| majority_baseline | – | 0.334448 | 0.000000 |
| tfidf_unigram | (1, 1) | 0.793887 | 0.818182 |
| tfidf_unigram_bigram | (1, 2) | 0.793970 | 0.797980 |

Vald textmodell: **tfidf_unigram_bigram**.
Urval: högst macro-F1, därefter negativ recall, därefter enskilda ord.
Urvalet använder orundade värden; tabellen visar sex decimaler.
Skillnad i macro-F1 mot baseline: +0.459522.
Macro-F1 ger de två klassernas F1 lika vikt. Negativ recall visar andelen
negativa recensioner som hittas. Odefinierade mått sätts till 0.

## Sparad modell och begränsningar

Vald pipeline sparas med joblib och återanvänds utan ny fit vid sluttestet.
Ingen omträning på train + validation görs. Testdata har inte lästs.
Validering har använts för modellval och är därför inte ett oberoende sluttest.
Resultaten gäller detta begränsade engelskspråkiga Yelp-underlag och visar
inte hur modellen fungerar på svenska eller på alla restaurangers omdömen.
Fler mått, confusion matrix och felanalys kompletteras i efterföljande uppgifter.
