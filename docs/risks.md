# Projektrisker

Riskerna utgår från datakällan, datakvalitetskontrollen, den låsta
modellutvärderingen och felanalysen i `reports/error_analysis.md`.
Ingen ny modell har tränats för den här genomgången.

## 1. Dataurval och datakvalitet

**Vad risken innebär.** Underlaget är 1 000 engelska meningar från UCI:s
Sentiment Labelled Sentences, bara Yelp-filen. Meningarna kommer från ett
äldre Yelp-material och valdes för att vara tydligt positiva eller negativa.
Efter dubbletthanteringen återstår 996 rader. Det finns ingen neutral klass,
och blandade omdömen var inte målet med urvalet. Fyra exakta dubbletter med
samma etikett har tagits bort. Inga etikettkonflikter fanns i filen.

**Möjlig konsekvens.** Testresultatet kan se mer användbart ut än det är för
verkliga recensioner. Felanalysen visar redan blandade meningar som modellen
läser fel, till exempel review_id 120 och 880. En restaurangansvarig som litar
på accuracy 0,745 kan missa klagomål eller agera på nöjda omdömen. Siffran
säger inget om svenska texter eller om ett annat restaurangdataset.

**Förebyggande.** Behåll källan, licensen och hashdokumentationen i
`docs/data_source.md`. Beskriv begränsningen tillsammans med varje resultat.
Använd inte testmåttet som bevis för en annan texttyp. Ändra inte den
befintliga uppdelningen för att få ett snyggare resultat.

**Reservplan.** Om modellen ska användas på svenska, på hela recensioner eller
på en annan restaurangs omdömen: stoppa automatisk klassning. Samla ett nytt
märkt underlag, dokumentera ursprung och urval, och gör en ny uppdelning med
samma regler som i `src/split_data.py` innan någon ny träning. Tills dess får
en människa läsa omdömena. De nuvarande testsiffrorna återanvänds inte som om
de gällde det nya underlaget.

## 2. Möjliga personuppgifter

**Vad risken innebär.** Datakvalitetskontrollen hittade inga e-postadresser,
telefonnummer, webbadresser eller sifferföljder med minst sex siffror. Den
hittade 4 möjliga namnträffar. Skolans krav gäller behandling av
personuppgifter i projektet, inte enbart publicering av exempel. Hantering av
de möjliga namnträffarna är inte avgjord. Gruppen behöver stämma av med läraren
innan fortsatt behandling. Inget godkännande finns dokumenterat.

Felanalysen citerar inte de träffarna. Namnkontrollen gav inga träffar i
testfilen, och exemplen i `reports/error_analysis.md` är korta utdrag utan
personnamn eller verksamhetsnamn. Råtexten finns ändå kvar lokalt i
datafilerna.

**Möjlig konsekvens.** Namn kan spridas via rapporter, demotext, loggar eller
versionshantering. Även en bearbetning som aldrig publiceras kan bryta mot
skolans krav. Om läraren senare kräver att träffarna tas bort blir de
nuvarande modellerna och uppdelningarna svåra att försvara oförändrade, eftersom
de är byggda på materialet som innehåller träffarna.

**Förebyggande.** Skriv inte ut markerade recensioner i rapporter. Lägg inte
rådata, bearbetade data eller modellfiler i Git. De är redan ignorerade.
Citera bara korta avidentifierade utdrag när en felanalys kräver text.
Fråga läraren innan träffarna behandlas vidare, och dokumentera svaret.

**Reservplan.** Om läraren säger att träffarna ska bort: ta bort de raderna
innan eventuell ny träning, dokumentera de nya antalen och träna om från den
rensade filen. Återanvänd inte de gamla testmåtten. Om läraren säger att
projektet ska avbrytas i väntan på besked: stoppa ny bearbetning av råfilen och
låt de redan skrivna rapporterna vara utan namn eller fullständiga
recensioner. Publicera inte exempel ur de fyra träffarna under tiden.

## 3. Modellens begränsningar i praktisk användning

**Vad risken innebär.** Den valda modellen är en TF-IDF-logistisk regression
för engelska och har två klasser. På den låsta testmängden har den accuracy
0,745, macro-F1 0,744841 och negativ recall 0,72. Den missar 28 av 100 negativa
recensioner och flaggar 23 av 100 positiva. Baseline har accuracy 0,5 och
negativ recall 0 eftersom den alltid väljer klass 1. Felanalysen visar att
ord som "very", "good" och "is" kan väga tyngre än ett svagt eller okänt
negativt ord, och att "not" kan fälla en positiv mening.

**Möjlig konsekvens.** Automatiska åtgärder kan dölja verkliga problem eller
skapa onödigt arbete kring nöjda gäster. Baseline kan se acceptabel ut om man
bara tittar på accuracy. Resultatet kan också feltolkas som en generell
restaurangmodell, trots att det bara gäller detta engelska meningstest.

**Förebyggande.** Rapportera macro-F1, negativ recall och confusion matrix
tillsammans med accuracy. Beskriv baseline som en jämförelse som inte hittar
klagomål. Använd textmodellen som stöd vid manuell genomgång, inte som enda
beslut. Ändra inte hyperparametrar eller modellval efter att testresultatet
har setts.

**Reservplan.** Om ett fel får praktisk betydelse, till exempel ett missat
hygienklagomål: låt en människa läsa originaltexten och ignorera prediktionen
för det beslutet. Om verksamheten behöver klassning av svenska eller blandade
helrecensioner: ta modellen ur bruk för det syftet och gör en ny utvärdering
på representativ text. Behåll den nuvarande modellen oförändrad som referens
för just detta dataset.

## 4. Reproducerbarhet

**Vad risken innebär.** Rådata, bearbetade data, uppdelningar och modellfiler
är lokala och ignoreras av Git. En annan dator har därför inte testfilen eller
`models/tfidf_logistic_regression.joblib` förrän flödet har körts.
`python-version.txt` anger Python 3.12.9. Felanalysen i den här arbetsytan
kördes med projektets `.venv`, som är Python 3.12.6, och med paketen i
`requirements.txt`, inklusive scikit-learn 1.9.1. Slumpfröet för uppdelning
och modell är 42.

**Möjlig konsekvens.** En körning med annan Python- eller paketversion kan ge
andra prediktioner. Då går det inte att kontrollera felanalysens 51
`review_id` mot en ny modell. Om någon ändrar uppdelningen eller tränar om
efter att ha sett testresultatet blandas modellval och sluttest.

**Förebyggande.** Kör från projektroten med `.\.venv\Scripts\python.exe` och
den låsta kravfilen. Återskapa data och modeller med
`src/run_pipeline.py` när de saknas lokalt. Låt `random_state=42` och den
sparade uppdelningen vara oförändrade. Dokumentera vilken Python-version som
faktiskt användes när ett resultat publiceras.

**Reservplan.** Om modellfilen eller `data/splits/test.csv` saknas: skapa inga
exempel på fel. Kör pipeline-skriptet och därefter `src/final_evaluation.py`.
Jämför den nya confusion matrix med `reports/metrics_test.csv` innan
felanalysen återanvänds. Om siffrorna skiljer sig, behåll båda resultaten märkta
med miljö och behandla inte de gamla exemplen som giltiga för den nya
modellen. Byt inte modell utifrån skillnaden.
