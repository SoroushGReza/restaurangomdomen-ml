# Användning av generativ AI

Generativ AI har använts som stöd under projektarbetet, bland annat för:

- planering och uppdelning av arbetsuppgifter,
- förslag på kod och automatiska tester,
- felsökning och kodgranskning,
- utkast till dokumentation och förklaringar av metoder och resultat.

## Kontroll av förslagen

Kodförslag har kontrollerats genom lokala körningar och automatiska tester.
Testerna omfattar bland annat majoritetsbaseline, matchning via review_id,
utvärderingsmått mot handräknade exempel, skydd mot dataläckage och
återanvändning av sparade modeller.

Dataflödet, modellträningen och demonstrationen har körts lokalt.
Vid den dokumenterade kontrollen passerade 60 tester.
Projektets resultat kommer från körningar på Yelp-underlaget.
Syntetiska exempel används för kodtester och demonstration, inte som
underlag för den redovisade testprestandan.

Ändringar hanteras genom commits och pull requests så att de kan granskas
och följas upp inom gruppen.