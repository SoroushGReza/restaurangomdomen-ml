# Datakvalitet

Bearbetningen utgår från Yelp-filen som `src/load_data.py` läser in.
Originaltexten behålls i kolumnen `text`. Normaliseringen nedan används
bara för att upptäcka dubbletter och ingår inte i modellens textbehandling.

## Inläsning och grundkontroller

Filen läses med `load_yelp_reviews`. Den kontrollen fångar en saknad fil,
en första rad som felaktigt blivit rubrik och ett radantal som inte stämmer
med originalfilen. Därefter kontrolleras varje rad.

| Kontroll | Antal rader |
| --- | --- |
| Inlästa rader | 1000 |
| Saknade värden | 0 |
| Tomma texter | 0 |
| Ogiltiga etiketter | 0 |

### Borttaget i grundkontrollen

Inga rader togs bort för tom text, saknat värde eller ogiltig etikett.

## Dubbletter

Jämförelsenormaliseringen gör tre saker med en kopia av texten:

- blanksteg i början och slutet tas bort
- upprepade blanksteg blir ett enda blanksteg
- skiftläge jämnas ut med `casefold`

Regler:

- Exakt samma text och samma etikett behålls en gång. Första `review_id` behålls.
- Exakt samma text med motstridiga etiketter tas bort helt och räknas nedan.
- Texter som blir lika först efter normaliseringen är nära dubbletter.
  Originaltexterna behålls och raderna får ett gemensamt `duplicate_group_id`.
- Rapporten redovisar `review_id`, etikett och orsak, inte recensionstexten.

### Exakta dubbletter med samma etikett

| review_id | etikett | orsak | förklaring |
| --- | --- | --- | --- |
| 815 | 1 | exakt dubblett med samma etikett | samma text som review_id 335 |
| 817 | 0 | exakt dubblett med samma etikett | samma text som review_id 384 |
| 844 | 0 | exakt dubblett med samma etikett | samma text som review_id 381 |
| 847 | 0 | exakt dubblett med samma etikett | samma text som review_id 506 |

### Identiska texter med motstridiga etiketter

Inga identiska texter med motstridiga etiketter hittades.

### Nära dubbletter

Inga textvarianter som bara skiljer sig i skiftläge eller blanksteg hittades.
Varje kvarvarande rad har därför ett eget `duplicate_group_id`, satt till radens `review_id`.

## Personuppgifter

Recensionerna kontrollerades efter e-postadresser, telefonnummer,
webbadresser och sifferföljder med minst sex siffror. Möjliga personnamn
markerades när ett förnamn står direkt vid en yrkesroll som server,
waitress, waiter eller manager, eller i formuleringarna `thanks to`
och `went above and beyond`.

| Kontroll | Träffar |
| --- | --- |
| E-post | 0 |
| Telefonnummer | 0 |
| Webbadress | 0 |
| Lång sifferföljd | 0 |
| Möjligt personnamn | 4 |

Skolans krav gäller behandling av personuppgifter i projektet, inte enbart
publicering av exempel. Träffarnas text skrivs inte ut i den här rapporten.
4 möjliga namnträffar har identifierats.
Gruppen behöver stämma av med läraren hur dessa ska hanteras innan fortsatt behandling.
Raderna ligger kvar i datafilerna tills den avstämningen är gjord.

## Kvarstående kvalitetsproblem

Inga tomma texter, saknade värden eller ogiltiga etiketter återstår. Klassfördelningen är 497 negativa och 499 positiva. Textlängden är minst 11, median 51 och högst 149 tecken. Stavning och skiljetecken från källan är kvar, eftersom originaltexten bevaras.

Hantering av de möjliga namnträffarna är inte avgjord.
Gruppen behöver stämma av med läraren innan fortsatt behandling.

## Sparat resultat

| Uppgift | Värde |
| --- | --- |
| Fil | `data/processed/reviews.csv` |
| Rader | 996 |
| Kolumner | `review_id`, `text`, `label`, `duplicate_group_id` |
| Negativa (0) | 497 |
| Positiva (1) | 499 |

Filen sparas lokalt. Innehållet i `data/processed/` är gitignorerat,
förutom `.gitkeep`.

Kör från projektroten:

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py
```
