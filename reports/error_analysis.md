# Felanalys av den sparade textmodellen

## Metod

Felanalysen använder den redan valda modellen `tfidf_unigram_bigram` i
`models/tfidf_logistic_regression.joblib` och den befintliga testfilen
`data/splits/test.csv`. Modellen laddades och kördes med `predict` på
kolumnen `text`. Inget anrop till `fit` gjordes. Etiketten jämfördes med
prediktionen via `review_id`.

Testfilen har 200 rader, 100 negativa (0) och 100 positiva (1). Det är samma
underlag som i `reports/metrics_test.csv`. Antalet fel stämmer med confusion
matrix för textmodellen: verklig 0 förutsades som 1 i 28 fall, och verklig 1
förutsades som 0 i 23 fall. Totalt 51 fel. 149 rader klassificerades rätt.

`predict` väljer klass 1 när sannolikheten för klass 1 är minst 0,5.
Sannolikheterna nedan kommer från den sparade modellens `predict_proba`.
Ordbidragen är TF-IDF-värde multiplicerat med koefficienten för klass 1 i den
sparade logistiska regressionen. Träningsräkningar kommer från
`data/splits/train.csv` och används bara för att tolka koefficienterna.
Uppdelningen, hyperparametrarna och modellvalet är oförändrade.

Namnkontrollen i `person_reference_ids` gav inga träffar i testfilen.
Utdragen är korta. Personnamn och verksamhetsnamn citeras inte.
Fullständig text finns bara i den lokala testfilen.

## Vad felen betyder

En **missad negativ recension** har verklig etikett 0 och prediktion 1.
Modellen kallar ett klagomål positivt. Det är de 28 cellerna där verklig klass
0 förutsades som 1.

En **felaktigt flaggad positiv recension** har verklig etikett 1 och
prediktion 0. Modellen kallar ett nöjt omdöme negativt. Det är de 23 cellerna
där verklig klass 1 förutsades som 0.

## Alla felklassificerade test-id

Missade negativa, 28 stycken:

`21, 80, 87, 120, 121, 142, 143, 147, 229, 371, 375, 388, 422, 489, 622, 646, 651, 732, 749, 755, 779, 820, 880, 912, 922, 939, 961, 989`

Felaktigt flaggade positiva, 23 stycken:

`49, 65, 97, 189, 228, 247, 272, 396, 437, 452, 477, 492, 553, 605, 662, 668, 721, 746, 747, 783, 788, 843, 878`

Fjorton av dessa gås igenom nedan. Övriga 37 räknas med i mönstren men citeras
inte, för att inte sprida mer recensionstext än som behövs för analysen.

## Genomgångna exempel

### Missade negativa recensioner

1. **review_id 142.** Verklig etikett 0, prediktion 1, sannolikhet för klass 1: 0,580.
   Utdrag: "wasn't helpful or friendly".
   "friendly" ger det starkaste positiva bidraget. "wasn" drar åt negativt, men
   svagare. Vektoriseraren behåller inte det ensamma t:et i "wasn't".
   Negationen finns i meningen, men det positiva ordet väger tyngre.

2. **review_id 622.** Verklig etikett 0, prediktion 1, sannolikhet 0,511.
   Utdrag: "Very disappointing!!!".
   "very" har koefficienten 0,851 och större positivt bidrag än
   "disappointing" har negativt. Bigrammet "very disappointing" finns inte i
   ordförrådet. Felet ligger precis över tröskeln.

3. **review_id 912.** Verklig etikett 0, prediktion 1, sannolikhet 0,604.
   Utdrag: "Very, very sad."
   "very" dominerar. "sad" finns i ordförrådet men har bara koefficienten
   -0,183. Texten är 15 tecken.

4. **review_id 21.** Verklig etikett 0, prediktion 1, sannolikhet 0,635.
   Utdrag: "The Burrittos Blah!".
   "blah" och "burrittos" saknas i ordförrådet. Det enda tydliga kända bidraget
   är "the", som har positiv koefficient. Texten är 19 tecken och felet är inte
   ett gränsfall.

5. **review_id 120.** Verklig etikett 0, prediktion 1, sannolikhet 0,532.
   Utdrag: "liked the look and sound" följt av att besöket var "disappointing".
   Meningen innehåller både beröm och besvikelse. "very" är det starkaste
   positiva bidraget och "disappointing" ett av de negativa. Felet ligger nära
   tröskeln.

6. **review_id 880.** Verklig etikett 0, prediktion 1, sannolikhet 0,699.
   Utdrag: "aren't as good" och "used to be amazing", därefter "doughy and
   flavorless".
   "good" och "amazing" är de starkaste positiva bidragen. "flavorless" och
   "doughy" saknas i ordförrådet. Detta är den högsta positiva sannolikheten
   bland de missade negativa recensionerna.

7. **review_id 922.** Verklig etikett 0, prediktion 1, sannolikhet 0,616.
   Utdrag: "This place is disgusting!".
   "disgusting" finns, men koefficienten är bara -0,146. Ordet förekommer en
   gång i träningsdata. "is" och "place" ger positiva bidrag.

8. **review_id 489.** Verklig etikett 0, prediktion 1, sannolikhet 0,643.
   Utdrag: "rarely a good sign".
   "good" är det starkaste bidraget. Formuleringen säger att något sällan är
   ett gott tecken, men modellen fångar ordet "good".

### Felaktigt flaggade positiva recensioner

9. **review_id 97.** Verklig etikett 1, prediktion 0, sannolikhet 0,487.
   Utdrag: "Will be back again!".
   "back" och "again" har negativa koefficienter, -0,295 och -0,199.
   I träningsdata står "back" i 21 negativa och 11 positiva recensioner, och
   "again" i 8 negativa och 5 positiva. Bigrammet "be back" är svagt positivt
   men räcker inte.

10. **review_id 272.** Verklig etikett 1, prediktion 0, sannolikhet 0,471.
    Utdrag: "You cant go wrong with any of the food here."
    "cant" och "wrong" saknas i ordförrådet. "go" har koefficienten -0,423 och
    är ett av de starkaste negativa bidragen.

11. **review_id 452.** Verklig etikett 1, prediktion 0, sannolikhet 0,463.
    Utdrag: "Food was so gooodd."
    "gooodd" saknas, så stavningen ger inte samma signal som "good". "was" drar
    åt negativt. Texten är 19 tecken.

12. **review_id 492.** Verklig etikett 1, prediktion 0, sannolikhet 0,460.
    Utdrag: "so much good stuff" och "could not decide".
    "not" ger det starkaste negativa bidraget och "good" det starkaste positiva.
    Negationen gäller att gästen inte kunde välja, inte att maten var dålig.

13. **review_id 65.** Verklig etikett 1, prediktion 0, sannolikhet 0,488.
    Utdrag: "We are so glad we found this place."
    "glad" saknas i ordförrådet. Det starkaste negativa bidraget är "we", med
    koefficienten -0,430. Felet ligger nära tröskeln.

14. **review_id 878.** Verklig etikett 1, prediktion 0, sannolikhet 0,387.
    Utdrag: "Go To Place for Gyros."
    Meningen är en kort rekommendation utan "good" eller "great". "go" och
    "for" är de starkaste negativa bidragen. Detta är ett av de säkrare felen
    bland de positiva recensionerna.

## Fel som lästs men inte citeras i längd

Alla 51 texter lästes. Följande används bara som stöd för mönstren, med samma
etikett och prediktion som i id-listan:

- review_id 80, missad negativ, sannolikhet 0,552. Starkaste positiva bidraget är "good", i en mening som samtidigt säger att besöket kändes bortkastat.
- review_id 422, missad negativ, sannolikhet 0,591. Klagomålet gäller hygien. Ordet för det beskrivna problemet saknas i ordförrådet.
- review_id 820, missad negativ, sannolikhet 0,526. Översvallande negativ formulering. Starkaste bidragen är "this" och "place", inte ett tydligt negativt värderingsord.
- review_id 989, missad negativ, sannolikhet 0,550. Sarkastiskt beröm för att stället inte har stängt. Starkaste bidragen är "is" och "really".
- review_id 396, felaktigt flaggad positiv, sannolikhet 0,372. Ovanlig och översvallande positiv formulering. "nothing" ger det starkaste negativa bidraget.
- review_id 788, felaktigt flaggad positiv, sannolikhet 0,434. En positiv slutsats inleds med "don't". "don" är det starkaste negativa bidraget.

## Observerade mönster

Följande är räknat eller avläst i den sparade modellen:

- 30 av 51 fel ligger inom 0,05 från tröskeln 0,5. 16 av 28 missade negativa har sannolikhet under 0,55. 14 av 23 felaktigt flaggade positiva har sannolikhet över 0,45. Många fel är alltså gränsfall. Det gäller inte alla: review_id 880 har 0,699 och review_id 396 har 0,372.
- Korta texter finns bland felen, men de förklarar inte felmängden. Medianlängden är 57 tecken för felen och 51 tecken för de rätta prediktionerna. Fem fel har högst 20 tecken. Bland de 149 rätta prediktionerna har 14 texter också högst 20 tecken.
- Åtta fel innehåller ett negationsliknande ord enligt en enkel sökning efter former som "not", "no", "never", "n't", "cant" och "won". Det är review_id 121, 142, 272, 492, 732, 755, 788 och 880. Bland de rätta prediktionerna innehåller 35 av 149 ett sådant ord. Negation är alltså inte vanligare bland felen än bland de rätta.
- I de exempel där negationen faktiskt ingår i ett fel är mekanismen konkret. I review_id 142 väger "friendly" tyngre än "wasn". I review_id 492 väger "not" tyngre än "good", fast negationen inte gäller matens kvalitet. I review_id 272 saknas "cant" och "wrong".
- Blandade omdömen är synliga i review_id 80, 120 och 880: beröm och klagomål står i samma mening, och positiva ord med hög vikt finns kvar.
- Ovanliga eller okända ord är synliga i review_id 21 ("blah", "burrittos"), 422, 452 ("gooodd") och 880 ("flavorless", "doughy"). De saknas i ordförrådet. "disgusting" och "disappointing" finns, men med svaga vikter, och "disappointing" förekommer bara två gånger i träningsdata.
- Funktionsord och förstärkare har egna vikter. "very" har koefficienten 0,851, "is" 0,930 och "the" 0,654. Det syns direkt i "Very disappointing!!!", "Very, very sad." och "This place is disgusting!".
- Tre lästa fel är underdrivna eller sarkastiska i formuleringen: review_id 489, 820 och 989. De har inte kodats som en andel av alla 51.

## Hypoteser som räkningen inte belägger

Att korta texter generellt orsakar felen är en hypotes. Den stöds inte av
medianlängden eller av att korta texter också finns bland de rätta
prediktionerna. Korta texter kan ändå bli fel när de saknar kända
värderingsord, vilket exemplen 21, 452 och 878 visar.

Att modellen generellt misslyckas på negation är också en hypotes. Åtta fel
innehåller negation, men sådana ord är vanligare bland de rätta prediktionerna.
Det som observerats är enstaka meningar där ett positivt ord, ett okänt ord
eller ett "not" utan den avsedda betydelsen avgör.

Ironi som huvudförklaring är en hypotes. Tre meningar är formulerade
sarkastiskt eller underdrivet. Övriga fel har inte granskats med en
ironi-etikett, så andelen är okänd.

## Betydelse för en restaurangansvarig

En missad negativ recension betyder att ett verkligt klagomål om till exempel
mat, service eller hygien ser positivt ut. I det här testet händer det i 28 av
100 negativa recensioner. Ansvarig får då ingen signal om problemet och kan
låta det fortsätta.

En felaktigt flaggad positiv recension betyder att ett nöjt omdöme ser ut som
ett klagomål. I det här testet händer det i 23 av 100 positiva recensioner.
Ansvarig kan lägga tid på ett problem som gästen inte har beskrivit, eller
få en sämre bild av verksamheten än omdömena ger.

Majoritetsbaseline förutsäger alltid klass 1. Den missar alla 100 negativa
testrecensioner och flaggar inga positiva. Accuracy 0,5 beror på att testet är
balanserat, inte på att baseline hittar klagomål. Textmodellen är bättre på
både macro-F1, negativ recall och accuracy, men felen ovan räcker för att den
inte ska styra åtgärder utan en människa.

## Begränsningar

Datasetet är ett litet och äldre urval: 1 000 engelska meningar från UCI:s
Sentiment Labelled Sentences, Yelp-delen, varav 996 återstår efter den
dokumenterade dubbletthanteringen. Meningarna valdes för att vara tydligt
positiva eller negativa, inte för att spegla hela recensioner med neutrala
eller blandade omdömen. Det finns ingen neutral klass.

Testresultatet visar hur den sparade modellen fungerar på just denna
testuppdelning. Det visar inte hur modellen fungerar på svenska recensioner,
på andra restaurangdataset, på hela recensioner eller på aktuell text från en
verksamhet. Valideringsdata användes redan för modellvalet och ingår inte här.
Den här genomgången är inte ett skäl att byta modell eller hyperparametrar.

## Rekommendation

Använd textmodellen bara som stöd när en människa sållar engelska meningar av
samma typ som datasetet. Låt den inte dölja, publicera eller besvara omdömen
automatiskt. På testmängden missar den 28 procent av de negativa recensionerna
och flaggar 23 procent av de positiva. Baseline är sämre för samma uppgift
eftersom den inte hittar några klagomål. För svenska recensioner eller ett
annat restaurangunderlag behövs en ny, separat utvärdering. Modell och
testuppdelning bör ligga kvar som de är.
