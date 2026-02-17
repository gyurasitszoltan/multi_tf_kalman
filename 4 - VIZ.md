# Mit érdemes megfigyelni a vizualizációkban

> 10 interaktív Plotly HTML chart részletes útmutatója.
> Minden chart az `output/` mappában található.

---

## 1. `filtered_states.html` — A szűrő fő outputja

4 subplot egymás alatt, közös x-tengely:

- **(1) BTC ár + μ̂:** Az ár és a szűrt pillanatnyi log hozam ráta együtt. A μ̂ előjele mutatja a mikro-trend irányát.
- **(2) μ̂ ± 1σ:** A szűrt hozam ráta konfidencia sávval. A sáv szélessége mutatja, mennyire biztos a becslés.
- **(3) μ̂̇ (momentum):** Zöld = pozitív (erősödő trend), piros = negatív (gyengülő trend). Amikor zöldből pirosba vált → trendforduló.
- **(4) μ̂̈ (gyorsulás):** Az előjelváltásai inflexiós pontok — a trend hajlása megváltozik.

**Mire figyelj:**
- Amikor μ̂ még pozitív, de μ̂̇ már negatív → a dokumentum szerint ez „profit taking" jelzés.
- Amikor μ̂ < 0 és μ̂̇ > 0 → az esés lassul, reversal közeleg.
- A ±1σ sáv szélessége nagy TF frissüléseknél (óránként, 4 óránként) hirtelen összeszűkül.

---

## 2. `returns_comparison.html` — Nyers vs szűrt hozamok TF-enként

6 subplot (egy minden TF-hez: 1m, 5m, 15m, 1h, 4h, 1d):

- Szürke = nyers log hozam (zajos, eredeti mérés)
- Kék = szűrt rekonstrukció (`H_τ · x̂` — a Kalman-szűrő visszavetítése az adott TF-re)

**Mire figyelj:**
- **1m subplot:** A szürke rendkívül zajos, a kék simított → itt látszik a szűrés ereje.
- **4h és 1d subplot:** Fordított helyzet — kevés a nyers pont (ritkán frissül), de a szűrt rekonstrukció folytonos. A szűrő „interpolál" a nagy TF mérések között.
- A lényeg: **a szűrő az összes TF-ből szintetizál**, nem csak egyből. A kék vonal minden subploton ugyanazt a latens állapotot tükrözi, csak különböző TF-re vetítve.

---

## 3. `kalman_gain_dynamics.html` — A szűrő „figyelmének" eloszlása

2 subplot:

- **Felső:** A K (Kalman gain) mátrix Frobenius normája időben. Vertikális vonalak jelzik, mikor frissülnek a nagy TF-ek.
- **Alsó:** 3 vonal — az egyes állapotkomponensek (μ, μ̇, μ̈) összesített gain-je.

**Mire figyelj:**
- A Frobenius norma **ugrásszerűen megnő**, amikor nagyobb TF mérés érkezik (5-percenként, 15-percenként, óránként stb.) → ez a „fűrészfog" minta.
- Az alsó panelen a μ̂̈ (gyorsulás) gain-je arányaiban nagyobb a nagy TF frissüléseknél → a gyorsulást főleg a nagy TF-ek „tanítják", mert a H mátrixban a gyorsulás együtthatója τ³-mal skálázódik.
- A legtöbb percben (amikor csak 1m mérés van) a K kicsi → a szűrő főleg a saját predikciójára támaszkodik.

---

## 4. `innovation_anomaly.html` — Piaci meglepetések

2 subplot:

- **Felső:** Normalizált innováció TF-enként (scatter, színkódolva). Minden pont egy „meglepetés" — mennyire tért el a mérés a szűrő várakozásától.
- **Alsó:** Mahalanobis-távolság (d_k) időben + χ² küszöbvonalak (95%, 99%).

**Mire figyelj:**
- Ha **egy pont minden TF-en** kiugrik → valódi piaci esemény (hír, likvidáció, flash crash).
- Ha **csak 1m-en** ugrik ki → valószínűleg zaj, bid-ask bounce, nem valódi esemény.
- Az alsó panel anomália pontjai (piros körök): érdemes összevetni a BTC árfolyammal — egybeesnek-e nagy mozgásokkal?
- 30 napon ~1700-1800 anomália tipikus — ez az összes lépés ~4%-a.

---

## 5. `covariance_evolution.html` — Bizonytalanság

3 vonal log skálán: P₀₀ (μ bizonytalanság), P₁₁ (μ̇), P₂₂ (μ̈).

**Mire figyelj:**
- **Fűrészfog minta:** A P lecsökken minden nagy TF frissülésnél (pl. óránként, 4 óránként erősen), majd lassan visszanő a predikciós lépések alatt. Ez a Kalman-szűrő természetes viselkedése.
- Ha a P **tartósan nagy** marad → a szűrő „elvesztette a fonalat" (pl. flash crash után, amikor az innováció extrém nagy). Néhány lépés alatt újra stabilizálódik.
- P₂₂ (gyorsulás) a legnagyobb bizonytalanság, P₀₀ (szint) a legkisebb → ez konzisztens azzal, hogy a magasabb deriváltakat nehezebb becsülni.
- Az összesített konfidencia (`1/tr(P)`) is látható — nagyobb = magabiztosabb szűrő.

---

## 6. `prediction_accuracy.html` — Előrejelző képesség

3×2 grid: 3 horizont (5m, 15m, 60m) × 2 nézet:

- **Bal oszlop (scatter):** Prediktált vs tényleges hozam. Az ideális a 45°-os szaggatott egyenes.
- **Jobb oszlop (idősor):** Predikció ± 95% CI sáv + tényleges hozam (fehér pontok).
- Metrikák a scatter-eken: RMSE, MAE, hit rate.

**Mire figyelj:**
- **Hit rate** a legfontosabb metrika: hány %-ban egyezik a prediktált irány (előjel) a ténylegessel? 50% fölött → a szűrőnek van prediktív ereje; 50% = véletlen.
- **5m horizont:** Viszonylag szoros scatter, magasabb hit rate → rövid távon jobb a predikció.
- **1h horizont:** Szétszórtabb scatter → hosszabb távon nő a bizonytalanság (ahogy várható).
- A **95% CI sáv** az idősoros nézetben: lefedi-e nagyrészt a tényleges hozamot? Ha igen, a szűrő jól kalibált. Ha sok pont kilóg → a szűrő alulbecsüli a bizonytalanságot.

---

## 7. `trend_dashboard.html` — Kompozit jelzés

2 subplot:

- **Felső:** Trend score vonal + BTC ár. Zöld háttér = pozitív trend, piros = negatív.
- **Alsó:** 3 normalizált komponens stacked area chart (μ̂, μ̂̇, μ̂̈ hozzájárulása).

**Mire figyelj:**
- A **zöld/piros háttér** azonnal mutatja a szűrő által érzékelt trendet. Érdemes összevetni az árral: a pozitív (zöld) zónákban valóban emelkedett-e az ár?
- Az alsó area chart-on figyeld, melyik komponens dominálja:
  - Ha **μ̂ dominál** stabilan → erős, egyirányú trend.
  - Ha **μ̂̈ nagy** → volatilis, gyorsan változó piac.
  - Ha a három komponens **ellentmond egymásnak** (pl. μ̂ pozitív, μ̂̇ negatív) → bizonytalan, átmeneti állapot.

---

## 8. `sensitivity_q.html` — ⭐ A legfontosabb kutatási chart

5 különböző q értékkel futtatott szűrő összehasonlítása:

- q = 10⁻¹⁰, 10⁻⁹, 10⁻⁸, 10⁻⁷, 10⁻⁶

**Mire figyelj:**
- **q = 10⁻¹⁰ (halvány):** Szinte egyenes vonal — a szűrő „nem hiszi el" a változásokat, túl sima.
- **q = 10⁻⁶ (élénk):** Szinte a nyers hozamot követi — a szűrő „kikapcsolt", nincs simítás.
- Az **optimális q** az, amelyik követi a trendet, de nem reagál minden zajra. Ez tipikusan 10⁻⁸ – 10⁻⁷ között van kriptóra.
- Ha az innováció variancia szisztematikusan nagyobb mint S_k → a q túl kicsi. Ha kisebb → túl nagy. Az optimális q-nál a kettő megegyezik.

---

## 9. `h_compare.html` — Elméleti validáció

3 subplot:

- **Felső:** μ̂ a continuous H-val vs discrete H-val (két vonal).
- **Középső:** A kettő közötti eltérés.
- **Alsó:** Innováció variancia mindkét módban.

**Mire figyelj:**
- Ha a két vonal **gyakorlatilag egybeesik** → a folytonos közelítés elég jó, nem kell a bonyolultabb diszkrét formula.
- Ha **eltérnek** (főleg a nagy TF frissüléseknél) → a diszkrét változat megbízhatóbb.
- Az eltérés mértéke növekszik a TF méretével (mert a Taylor-közelítés maradéktagja τ³-mal skálázódik).
- Ez a chart megválaszolja: érdemes-e a diszkrét formulát használni, vagy a continuous is elég?

---

## 10. `smoother_rts.html` — Online vs utólagos

3 subplot:

- **Felső:** μ̂ online (szűrt, kék) vs μ̂ simított (RTS, narancssárga).
- **Középső:** μ̂̇ online vs simított.
- **Alsó:** P₀₀ online vs simított.

**Mire figyelj:**
- Az **RTS simított** simább és „előrelátóbb" — visszamenőleg felhasználja a jövőbeli méréseket is. Inflexiós pontokat korábban/pontosabban azonosít.
- Az alsó panel: a simított P **mindig ≤** az online P → ez matematikailag garantált (több információ = kisebb bizonytalanság). Ha eltérést látsz ettől, az bug.
- **Backtesting-hez** az RTS az ideális (utólagos elemzés, stratégia kiértékelés).
- **Élő döntéshez** csak az online szűrt érhető el (nincs jövőbeli adat).
- A két vonal közötti különbség mutatja, mennyit „ér" a jövőbeli információ — ha nagy a különbség, a szűrő gyakran „meglepődik".

---

## Összefoglaló: melyik chart mire ad választ?

| Kérdés | Chart |
|--------|-------|
| Működik-e a multi-TF szintézis? | `returns_comparison.html` + `kalman_gain_dynamics.html` |
| Mi az optimális q paraméter? | `sensitivity_q.html` |
| Van-e prediktív ereje a szűrőnek? | `prediction_accuracy.html` (hit rate) |
| Mikor volt anomália a piacon? | `innovation_anomaly.html` |
| Mennyire biztos a szűrő? | `covariance_evolution.html` |
| Melyik H mátrix a jobb? | `h_compare.html` |
| Mit nyer az RTS simítás? | `smoother_rts.html` |
| Mi a jelenlegi trend? | `trend_dashboard.html` + `filtered_states.html` |
