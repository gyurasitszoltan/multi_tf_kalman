# Kálmán-szűrő alkalmazása: Egyetlen coin log hozama több timeframe-en

> Eszmefuttatás — A log hozamok hierarchikus aggregációs struktúrájának kihasználása Kálmán-szűrővel.

---

## Tartalomjegyzék

1. [Motiváció és alapötlet](#1-motiváció-és-alapötlet)
2. [A log hozamok TF-ek közötti matematikai kapcsolata](#2-a-log-hozamok-tf-ek-közötti-matematikai-kapcsolata)
3. [Az állapottér felépítése](#3-az-állapottér-felépítése)
4. [A Kálmán-szűrő komponensei](#4-a-kálmán-szűrő-komponensei)
5. [A szűrő futtatása](#5-a-szűrő-futtatása)
6. [Kereskedési jelzések kinyerése](#6-kereskedési-jelzések-kinyerése)
7. [Gyakorlati megfontolások](#7-gyakorlati-megfontolások)

---

## 1. Motiváció és alapötlet

### 1.1 Miért vizsgáljunk log hozamot több TF-en?

Egy coin árfolyamának log hozama különböző timeframe-eken nem független megfigyelések halmaza — a TF-ek között **determinisztikus aggregációs kapcsolat** áll fenn. Az 1h-s log hozam matematikailag az 1m-es log hozamok összege. Ez azt jelenti, hogy a különböző TF-ek log hozamai **redundáns, de különböző zajszintű** mérései ugyanannak az alapfolyamatnak.

Ez a felismerés teszi a Kálmán-szűrőt különösen hatékonnyá: a szűrő képes a redundáns, de eltérő zajszintű információkat optimálisan kombinálni, és ezzel egy olyan hozambecslést előállítani, ami **bármelyik egyedi TF-nél pontosabb**.

### 1.2 Miben különbözik ez a multi-TF RSI esettől?

| Tulajdonság | Multi-TF RSI (III. eset) | Multi-TF log hozam (IV. eset) |
|---|---|---|
| Megfigyelt mennyiség | RSI (indikátor, 0–100 közé korlátozott) | Log hozam (korlátlan, additív) |
| TF-ek közötti kapcsolat | Heurisztikus (vonzás, áramlás) | Matematikailag egzakt (aggregáció) |
| Előfeldolgozás | Logit transzformáció szükséges | Nincs szükség transzformációra |
| Linearitás | Közelítőleg lineáris | Természetesen lineáris |
| Fő alkalmazás | Dashboard / döntéstámogatás | Precíz szűrt hozam- és momentum-becslés |

A log hozamok additív tulajdonsága miatt ez az eset **természetesebben illeszkedik** a Kálmán-szűrő lineáris keretrendszerébe, mint az RSI eset.

### 1.3 Az alapötlet vizuálisan

```
Árfolyam:    ████████████████████████████████████████████
             ↓          ↓              ↓              ↓
1m hozam:    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   (zajos, gyors)
5m hozam:    ████████  ████████  ████████  ████████     (simább, lassabb)
15m hozam:   ████████████████████████  ████████████     (még simább)
1h hozam:    ████████████████████████████████████████   (legsimább)
                                                        
             ↓ Kálmán-szűrő kombinálja mindet ↓
                                                        
Szűrt becslés: ═══════════════════════════════════════  (optimális)
```

---

## 2. A log hozamok TF-ek közötti matematikai kapcsolata

### 2.1 Aggregációs azonosság

Legyen `P_t` az árfolyam a `t` időpillanatban. Az 1 perces log hozam:

```
r_1m,t = log(P_t / P_{t-1min})
```

Az 5 perces log hozam:

```
r_5m,t = log(P_t / P_{t-5min}) = Σ_{i=0}^{4} r_1m,t-i
```

Általánosan, ha a `τ` timeframe `n_τ` darab alapegységből (legkisebb TF-ből) áll:

```
r_τ,t = Σ_{i=0}^{n_τ - 1} r_base,t-i
```

Ez **egzakt matematikai azonosság**, nem közelítés.

### 2.2 Az aggregáció hatása a zajra

Ha az 1m-es hozam varianciája `σ²_1m`, és az 1m-es hozamok időben közelítőleg függetlenek (gyenge feltevés), akkor:

```
Var(r_5m) ≈ 5 · σ²_1m
Var(r_15m) ≈ 15 · σ²_1m
Var(r_1h) ≈ 60 · σ²_1m
```

De a **jel-zaj arány** javul, mert az igazi trend-komponens lineárisan skálázódik, míg a zaj csak négyzetgyökkel:

```
SNR_τ ∝ √n_τ
```

Tehát a nagyobb TF hozama jobb jel-zaj aránnyal rendelkezik, de ritkábban frissül. A Kálmán-szűrő éppen ezt a tradeoff-ot kezeli optimálisan.

### 2.3 A hozam felbontása komponensekre

A pillanatnyilag megfigyelt hozamot felbonthatjuk:

```
r_τ,t = μ_τ,t + ε_τ,t
```

Ahol:
- `μ_τ,t` — az igazi, latens trend-komponens (amit becsülni akarunk)
- `ε_τ,t` — piaci mikrozaj, bid-ask bounce, likviditási zaj

A Kálmán-szűrő feladata: `μ_τ,t` és annak deriváltjainak optimális becslése az összes TF méréséből együttesen.

---

## 3. Az állapottér felépítése

### 3.1 A latens állapot: az alapfolyamat

Az alapötlet: létezik egy **latens, folytonos hozamfolyamat** `μ(t)`, ami az árfolyam valódi, zajmentes pillanatnyi log hozam rátája. Ebből deriváljuk az állapotot.

### 3.2 Állapotvektor definíció

```
x_k = [ μ_k, μ̇_k, μ̈_k ]ᵀ
```

Ahol:
- `μ_k` — latens pillanatnyi log hozam ráta (szint)
- `μ̇_k` — a hozam ráta változási sebessége (trend-momentum)
- `μ̈_k` — a hozam ráta gyorsulása (trend-gyorsulás)

Ez egy **3-dimenziós** állapotvektor. Ez lényegesen kompaktabb, mint a multi-TF RSI eset (10 dimenzió), mert itt a különböző TF-ek nem külön állapotok, hanem **ugyanannak a latens folyamatnak különböző időskálájú aggregációi**.

### 3.3 Miért elegendő 3 dimenzió?

Az állapotvektor `[μ, μ̇, μ̈]` egy másodrendű Taylor-közelítés a latens hozamfolyamatra:

```
μ(t + δ) ≈ μ(t) + μ̇(t)·δ + ½·μ̈(t)·δ²
```

Ezt az állapotot használjuk arra, hogy **bármelyik TF-re** előrejelzést adjunk — a TF-ek közötti különbség csak az integráció időtartamában van.

### 3.4 Opcionálisan bővített állapotvektor

Ha a volatilitás dinamikáját is modellezni akarjuk:

```
x_k = [ μ_k, μ̇_k, μ̈_k, σ²_k, σ̇²_k ]ᵀ
```

Ahol:
- `σ²_k` — pillanatnyi variancia (volatilitás²)
- `σ̇²_k` — a variancia változási rátája

Ezzel a szűrő a volatilitást is adaptívan becsüli, ami a Q és R mátrixok frissítéséhez használható. Az egyszerűség kedvéért a továbbiakban a 3-dimenziós alapesettel dolgozunk.

---

## 4. A Kálmán-szűrő komponensei

### 4.1 Állapotátmeneti mátrix (F)

A latens hozamfolyamat dinamikája (konstans gyorsulás modell):

```
         μ     μ̇     μ̈
F =  [   1     Δt    ½Δt²  ]
     [   0     1     Δt     ]
     [   0     0     1      ]
```

Ez azt mondja:
- A hozam ráta a momentum szerint változik
- A momentum a gyorsulás szerint változik
- A gyorsulás „random walk"-ot követ (a zaj a Q-ban van)

`Δt` az alapfrissítési időlépés (tipikusan 1 perc).

### 4.2 Megfigyelési mátrix (H) — a kulcselem

Ez a mátrix az, ami **a különböző TF-ek méréseit összeköti egyetlen latens állapottal**. Minden TF hozama a latens folyamat időbeli integrálja, tehát:

Az `τ` TF-hez tartozó mért log hozam a `[t-τ, t]` intervallum feletti integrál közelítése:

```
r_τ,t ≈ μ_k · τ + μ̇_k · ½τ² + μ̈_k · ⅙τ³
```

Tehát az `τ` TF-hez tartozó H sor:

```
H_τ = [ τ,   ½τ²,   ⅙τ³ ]
```

5 timeframe (1m, 5m, 15m, 60m, 240m) esetén:

```
         μ       μ̇         μ̈
H =  [  1      0.5       0.167    ]   — 1m (τ=1)
     [  5      12.5      20.83    ]   — 5m (τ=5)
     [  15     112.5     562.5    ]   — 15m (τ=15)
     [  60     1800      36000    ]   — 1h (τ=60)
     [  240    28800     2304000  ]   — 4h (τ=240)
```

A H mátrix sorai mutatják, hogy a nagyobb TF hozamai **sokkal érzékenyebbek** a μ̇ és μ̈ komponensekre — ez a TF-ek közötti „információs munkamegosztás":
- Az 1m hozam főleg μ-t méri (szint)
- A 4h hozam főleg μ̈-t méri (gyorsulás), mert az integrálás felgyűjti

### 4.3 Alternatív H mátrix: diszkrét aggregáció

Ha nem a folytonos integrálos közelítést, hanem a diszkrét aggregációt akarjuk modellezni (ami egzaktabb), a H mátrix a következőképpen épül fel.

Az `n`-perces TF log hozama `n` darab 1 perces hozam összege. Ha az 1 perces hozamot a latens állapotból a következőképpen közelítjük:

```
r_1m,t-i ≈ μ_{k} + μ̇_{k} · (−i) + ½ · μ̈_{k} · i²
```

Akkor az n perces aggregált hozam:

```
r_nm,k = Σ_{i=0}^{n-1} r_1m,t-i ≈ n·μ_k - μ̇_k · n(n-1)/2 + μ̈_k · n(n-1)(2n-1)/12
```

Tehát:

```
H_n = [ n,   -n(n-1)/2,   n(n-1)(2n-1)/12 ]
```

Az 5 TF-re:

```
         μ         μ̇           μ̈
H =  [  1        0            0             ]   — 1m  (n=1)
     [  5       -10           10            ]   — 5m  (n=5)
     [  15      -105          490           ]   — 15m (n=15)
     [  60      -1770         34810         ]   — 1h  (n=60)
     [  240     -28680        2292840       ]   — 4h  (n=240)
```

A negatív előjelek a μ̇ oszlopban azt tükrözik, hogy ha a momentum pozitív (emelkedő trend), a régebbi perces hozamok kisebbek voltak → az aggregált hozam kevesebb, mint `n·μ_k`, mert a korábbi hozamok még alacsonyabbak voltak.

### 4.4 Mérési zaj kovariancia (R)

A mérési zaj két forrásból áll:
- **Piaci mikrozaj** (bid-ask bounce, tick zaj): `σ²_micro`
- **Aggregációs zaj** (az n perces ablak nem pontosan illeszkedik a latens dinamikára): skálázódik `n`-nel

```
Var(ε_τ) ≈ n_τ · σ²_micro + σ²_aggreg(n_τ)
```

Ha egyszerűsítünk és azt feltételezzük, hogy az 1 perces mikrozajok közelítőleg függetlenek:

```
R = diag(σ²_1m, σ²_5m, σ²_15m, σ²_1h, σ²_4h)
```

Ahol:

```
σ²_nm ≈ n · σ²_1m
```

Tehát:

```
R ≈ σ²_1m · diag(1, 5, 15, 60, 240)
```

Ha a TF-ek mérési zajai korrelálnak (ami várható, mert a 15m hozam „tartalmazza" az 1m és 5m hozamok egy részét), az R mátrixnak nem-diagonális elemei is vannak:

```
R_ij = Cov(ε_τi, ε_τj)
```

Ez a **beágyazott TF-ek átfedéséből** adódik. Konkrétan:

```
Cov(r_5m, r_15m) ≠ 0
```

mert az 5 perces hozam teljes egészében benne van a 15 perces hozamban. A pontos kovariancia:

```
Cov(r_τi, r_τj) = min(n_τi, n_τj) · σ²_1m     ha τi befér τj-be
```

Tehát a teljes R mátrix:

```
R = σ²_1m · [ 1    1    1    1     1    ]
             [ 1    5    5    5     5    ]
             [ 1    5    15   15    15   ]
             [ 1    5    15   60    60   ]
             [ 1    5    15   60    240  ]
```

Ez egy fontos finomság: a szűrő tudja, hogy a nagy TF mérés „tartalmazza" a kis TF-et, és nem számítja kétszer.

### 4.5 Folyamatzaj kovariancia (Q)

A Q mátrix a latens hozamfolyamat kiszámíthatatlanságát kódolja. Egy standard „piecewise constant acceleration" modellben:

```
Q = q · [ Δt⁵/20    Δt⁴/8     Δt³/6  ]
        [ Δt⁴/8     Δt³/3     Δt²/2  ]
        [ Δt³/6     Δt²/2     Δt     ]
```

Ahol `q` a gyorsulás spektrális sűrűsége (skálár paraméter). Ez azt modellezi, hogy az igazi gyorsulás „random walk"-ot követ `q` intenzitással.

**A `q` paraméter hangolása:** nagyobb `q` → a szűrő gyorsabban reagál a változásokra (de zajérzékenyebb). Kisebb `q` → simább becslés (de lassabban követ). Kriptópiacon tipikusan viszonylag nagy `q` kell a gyors ármozgások miatt.

### 4.6 Kezdeti állapot

```
x̂_0|0 = [ r̄_1m,    0,    0 ]ᵀ
```

Ahol `r̄_1m` az első néhány perces hozam átlaga. A momentum és gyorsulás 0-ról indul (nincs információ).

```
P_0|0 = diag(σ²_init_μ,  σ²_init_μ̇,  σ²_init_μ̈)
```

Legyen nagy (pl. `P_0 = 100·I`), hogy a szűrő gyorsan alkalmazkodjon.

---

## 5. A szűrő futtatása

### 5.1 A frissítési séma — heterogén mintavételezés

Kritikus gyakorlati kérdés: a különböző TF-ek **különböző gyakorisággal** frissülnek.

```
1m:   minden percben új mérés
5m:   5 percenként új mérés
15m:  15 percenként új mérés
1h:   óránként új mérés
4h:   4 óránként új mérés
```

Két megközelítés:

#### a) Szekvenciális frissítés (ajánlott)

A szűrőt az **alapfrekvencián** (1 percenként) futtatjuk. Minden percben:

- Predikciós lépés: mindig végrehajtjuk
- Korrekciós lépés: **csak az éppen elérhető TF-ek méréseivel**

Ha `t mod 5 ≠ 0`, akkor az 5m mérés nem elérhető. Ilyenkor a H mátrixot és R mátrixot **dinamikusan szűkítjük**:

```
t = 1 perc:   H = H_1m (1×3),  R = R_1m (1×1)     — csak 1m elérhető
t = 5 perc:   H = [H_1m; H_5m] (2×3),  R = R_1m5m (2×2)  — 1m + 5m
t = 15 perc:  H = [H_1m; H_5m; H_15m] (3×3)        — 1m + 5m + 15m
t = 60 perc:  H = [H_1m; ...; H_1h] (4×3)           — 1m + 5m + 15m + 1h
t = 240 perc: H = H_full (5×3)                       — mind az 5 TF
```

A szűrő egyenletei automatikusan kezelik a változó méretű H-t — a K nyereség is változó méretű lesz.

Ez a megközelítés pontosan kódolja az információ aszimmetrikus érkezését.

#### b) Batch frissítés

Alternatívaként várhatsz az alap-TF frissítéséig (pl. 5 percenként) és egyszerre adod be az összes elérhető mérést. Egyszerűbb, de elveszíted az 1 perces felbontást.

### 5.2 A teljes algoritmus (szekvenciális frissítéssel)

Minden `k` percben:

```
PREDIKCIÓS LÉPÉS (mindig):
    1.  x̂_k|k-1 = F · x̂_{k-1|k-1}
    2.  P_k|k-1  = F · P_{k-1|k-1} · Fᵀ + Q

MÉRÉSI VEKTOR ÖSSZEÁLLÍTÁS:
    3.  Határozd meg, mely TF-ek elérhetők: S = {τ : k mod n_τ == 0}
    4.  z_k = [ r_τ,k  |  τ ∈ S ]              — csak az elérhető TF-ek mérései
    5.  H_k = [ H_τ    |  τ ∈ S ] sorokból      — a megfelelő H sorok
    6.  R_k = R[S, S]                            — a megfelelő R részmátrix

KORREKCIÓS LÉPÉS:
    7.  ỹ_k = z_k - H_k · x̂_k|k-1
    8.  S_k = H_k · P_k|k-1 · H_kᵀ + R_k
    9.  K_k = P_k|k-1 · H_kᵀ · S_k⁻¹
   10.  x̂_k|k = x̂_k|k-1 + K_k · ỹ_k
   11.  P_k|k = (I - K_k · H_k) · P_k|k-1

OUTPUT:
   12.  μ̂_k = x̂_k|k[0]      — szűrt pillanatnyi hozam ráta
   13.  μ̂̇_k = x̂_k|k[1]      — szűrt momentum
   14.  μ̂̈_k = x̂_k|k[2]      — szűrt gyorsulás
   15.  conf_k = 1/tr(P_k|k)  — konfidencia
```

### 5.3 A Kálmán-nyereség viselkedése

A nyereség dinamikusan változik a rendelkezésre álló mérések függvényében:

**1 percenként (csak 1m mérés):** A K kicsi és főleg μ-t korrigálja, mert egyetlen zajos mérésből nem lehet μ̇-t és μ̈-t jól becsülni.

**5 percenként (1m + 5m mérés):** A K nagyobb, és a két TF együttes információja már lehetővé teszi a μ̇ becslésének érdemi korrekcióját.

**Óránként (1m + 5m + 15m + 1h):** A K a legerősebb, mind a 3 állapotkomponenst érdemileg korrigálja.

Ez azt jelenti, hogy a szűrt becslés **óránként ugrásszerűen pontosabbá válik** (amikor a nagy TF mérése is beérkezik), miközben a köztes percekben a predikció tartja fenn a folytonosságot.

---

## 6. Kereskedési jelzések kinyerése

### 6.1 Szűrt pillanatnyi hozam ráta (μ̂)

```
μ̂_k = x̂_k|k[0]
```

Ez az alapjel. A zajos 1m-es hozam helyett egy optimálisan szűrt értéket kapsz, ami az összes TF információját tartalmazza.

**Értelmezés:**
- `μ̂ > 0` és stabil → bullish mikro-trend
- `μ̂ < 0` és stabil → bearish mikro-trend
- `μ̂ ≈ 0` → semleges / range-piac

### 6.2 Szűrt momentum (μ̂̇)

```
μ̂̇_k = x̂_k|k[1]
```

A hozam ráta **változásának** becslése — ez mutatja, hogy a trend erősödik vagy gyengül.

**Értelmezés:**
- `μ̂ > 0` és `μ̂̇ > 0` → a trend erősödik (long tartás / növelés)
- `μ̂ > 0` és `μ̂̇ < 0` → a trend lassul (profit taking megfontolása)
- `μ̂ < 0` és `μ̂̇ > 0` → az esés lassul (reversal közeleg?)
- `μ̂ < 0` és `μ̂̇ < 0` → az esés gyorsul (risk-off)

### 6.3 Szűrt gyorsulás (μ̂̈)

```
μ̂̈_k = x̂_k|k[2]
```

A „gyorsulás a gyorsulásban" — a trend hajlékonyságának mértéke.

**Értelmezés:**
- `μ̂̈` előjel váltása → inflexiós pont a trend dinamikájában
- `μ̂̈ → 0` tartósan → állandó momentumú trend (lineáris ármozgás)
- `|μ̂̈|` nagy → a piac gyorsan változik → magasabb kockázat

### 6.4 Trend-erő kompozit jel

Kombináljuk mindhárom komponenst egyetlen „trend-erő" mutatóvá:

```
trend_score_k = w₁ · μ̂_k / σ_μ + w₂ · μ̂̇_k / σ_μ̇ + w₃ · μ̂̈_k / σ_μ̈
```

Ahol `σ_μ, σ_μ̇, σ_μ̈` normalizáló tényezők (gördülő ablak szórások), és `w₁ + w₂ + w₃ = 1`. Tipikus súlyozás:

```
w₁ = 0.5    — a szint a legfontosabb
w₂ = 0.35   — a momentum erős jel
w₃ = 0.15   — a gyorsulás kiegészítő
```

### 6.5 Prediktív hozambecslés

A szűrt állapotból tetszőleges horizontra előrejelezhetünk:

```
r̂_{t→t+τ} = μ̂_k · τ + μ̂̇_k · ½τ² + μ̂̈_k · ⅙τ³
```

Például 5 perces előrejelzés (τ = 5):

```
r̂_5m = 5·μ̂ + 12.5·μ̂̇ + 20.83·μ̂̈
```

A predikció bizonytalansága:

```
Var(r̂_{t→t+τ}) = H_τ · P_k|k · H_τᵀ
```

Ez azonnal **konfidencia-intervallumot** ad az előrejelzéshez:

```
CI_95% = r̂ ± 1.96 · √Var(r̂)
```

### 6.6 Innováció-alapú anomália-detekció

```
d_k = ỹ_kᵀ · S_k⁻¹ · ỹ_k
```

Ahol `ỹ_k` az innováció vektor, `S_k` az innováció kovariancia.

A `d_k` χ² eloszlást követ `p` szabadsági fokkal (p = elérhető mérések száma az adott lépésben).

```
p = 1 (csak 1m mérés):   küszöb_95% = 3.84
p = 2 (1m + 5m):          küszöb_95% = 5.99
p = 3 (1m + 5m + 15m):    küszöb_95% = 7.81
p = 5 (mind az 5 TF):     küszöb_95% = 11.07
```

Ha `d_k > küszöb` → a piac szignifikánsan eltér a modell-predikciótól → anomália → potenciális kereskedési esemény.

### 6.7 TF-specifikus reziduumok elemzése

Az innováció vektor egyes komponensei külön-külön is értékesek:

```
ỹ_τ,k = r_τ,k^mért - H_τ · x̂_k|k-1
```

**Normalizált reziduum TF-enként:**

```
ỹ_τ,k^norm = ỹ_τ,k / √S_k[τ,τ]
```

Ha `|ỹ_1m^norm| >> |ỹ_1h^norm|` → a meglepetés csak a kis TF-en van → valószínűleg zaj, nem valódi esemény.

Ha `|ỹ_1h^norm| >> |ỹ_1m^norm|` → a meglepetés a nagy TF-en is megjelenik → valódi piaci esemény, a szűrő erősen korrigálni fog.

### 6.8 Kálmán simított (smoothed) becslés

A fenti szűrő „online" (forward-only). Ha néhány perc késleltetés elfogadható (pl. nem HFT, hanem manuális kereskedés), a Rauch–Tung–Striebel (RTS) simítóval **visszamenőleg is pontosíthatjuk** a becslést:

```
Visszafelé haladva k = K-1, K-2, ..., 0:
    C_k     = P_k|k · Fᵀ · P_{k+1|k}⁻¹
    x̂_k|K  = x̂_k|k + C_k · (x̂_{k+1|K} - x̂_{k+1|k})
    P_k|K  = P_k|k + C_k · (P_{k+1|K} - P_{k+1|k}) · C_kᵀ
```

A simított becslés (`x̂_k|K`) pontosabb, mint a szűrt (`x̂_k|k`), mert a jövőbeli méréseket is felhasználja. Ez hasznos a kereskedési napló utólagos elemzéséhez és a stratégia backtest-eléséhez.

---

## 7. Gyakorlati megfontolások

### 7.1 A q paraméter hangolása

A `q` (gyorsulás spektrális sűrűsége) a szűrő egyetlen kritikus hangolási paramétere. Hatása:

| q érték | Hatás | Alkalmazás |
|---|---|---|
| Nagyon kicsi | Ultra-sima becslés, lassú reakció | Hosszú távú trend-detekció |
| Kicsi | Sima, mérsékelt reakció | Swing trading asszisztens |
| Közepes | Kiegyensúlyozott | Általános célú szűrés |
| Nagy | Gyors reakció, zajérzékeny | Scalping, HFT asszisztens |
| Nagyon nagy | Szinte nincs simítás | A szűrő „kikapcsol" |

**Optimális q becslése:** Innovation-based estimation. Ha a szűrő jól van paraméterezve:

```
E[ỹ_k · ỹ_kᵀ] ≈ S_k
```

Ha az empirikus innováció variancia szisztematikusan > S_k → q túl kicsi (a szűrő túl lassú, nem hiszi el a változásokat).
Ha az empirikus innováció variancia < S_k → q túl nagy (a szűrő túl „idegesen" reagál).

### 7.2 A modell korlátai

**Linearitás:** A Kálmán-szűrő lineáris dinamikát feltételez. A kriptópiac nem lineáris — hirtelen likvidációs kaszkádok, stop-loss láncreakciók, és funding rate ugrások megjelenhetnek. Ilyenkor a szűrő rövid ideig „elveszíti a fonalat" (nagy P, nagy innováció), majd néhány lépés alatt újra behangolódik.

**Gaussi zaj:** A Kálmán-szűrő Gauss-zajt feltételez. A kriptó hozamok vastag farkú (fat-tailed) eloszlásúak. A szűrő ettől még működik (a legjobb lineáris becslést adja), de az anomália-detekció küszöbértékei nem egészen pontosak. Robusztusabb alternatíva: Student-t eloszlás alapú szűrő.

**Stacionaritás:** Az F mátrix konstans, de a valódi piaci dinamika változik. Megoldás: gördülő ablakos újrabecslés, vagy adaptív Q.

### 7.3 Kombináció más jelzésekkel

A szűrő outputja önmagában nem kereskedési stratégia, hanem egy **szűrt jelkészlet**, amit kombinálhatsz:

- **Order flow adatokkal:** A szűrt momentum + a LOB (limit order book) egyensúlytalansága együtt erősebb jel.
- **Funding rate-tel:** Ha a szűrt momentum bullish, de a funding rate extrém magas → short squeeze kockázat → óvatosság.
- **Multi-coin Kálmán-szűrővel (II. eset):** A szűrt momentum egy coinra + a cross-coin lead-lag struktúra → erősebb predikció.
- **Multi-TF RSI szűrővel (III. eset):** Két különböző szűrő összhangja → magasabb konfidencia.

### 7.4 Backtest-figyelmeztetés

A Kálmán-szűrő paramétereinek (q, R, F csatolási paraméterei) optimalizálása historikus adaton **overfitting kockázatot** hordoz. Javasolt:

- Walk-forward validáció: hangold a paramétereket az első `N` perces adaton, teszteld a következő `M` percen, gördítsd tovább.
- Out-of-sample tesztelés: soha ne kereskedj a paraméterek becsléséhez használt adaton.
- Robusztusság-ellenőrzés: perturbáld a paramétereket ±20%-kal — ha a stratégia szétesik, túl érzékeny.

### 7.5 Összefoglalás: miért erős ez a megközelítés?

A multi-TF log hozam Kálmán-szűrő ereje abban rejlik, hogy:

1. **Egyetlen kompakt modellben** (3 állapot) szintetizálja az összes TF információját.
2. **Automatikusan súlyozza** a TF-eket megbízhatóságuk szerint (a nagy TF simább, tehát a szűrő jobban „hisz" neki).
3. **Prediktív**: nemcsak szűrt becslést ad, hanem előrejelzést is konfidencia-intervallummal.
4. **Anomália-detektáló**: a modell-várakozástól való eltérés azonnal mérhető.
5. **A latens állapot fizikailag értelmes**: μ = „hol tartunk", μ̇ = „merre megyünk", μ̈ = „gyorsulunk vagy lassulunk".

Ez nem fekete doboz — minden output értelmezhető, és a kereskedő döntéstámogatásához természetesen illeszkedik.

---

### Figyelmeztetés

Ez a dokumentum kizárólag oktatási és elméleti célokat szolgál. Nem minősül befektetési tanácsnak. A szűrő-alapú kereskedési stratégiák nem garantálnak profitot. A kriptovaluták rendkívül volatilis eszközök. Minden befektetési döntés a felhasználó saját felelőssége.

---

*Generálva: 2026. február 17.*
