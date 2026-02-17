# Paraméterek részletes magyarázata

> A `config.yaml` Kalman és Trend szekciójának minden paramétere.

---

## Kalman paraméterek

### `q: 1e-8` — A szűrő egyetlen igazi hangológombja

Ez a **gyorsulás spektrális sűrűsége** — azt mondja meg, mennyire „bízik" a szűrő a saját predikciójában vs. az új mérésekben.

A Q mátrix teljes egészében ebből az egyetlen számból épül:

```
Q = q · [Δt⁵/20  Δt⁴/8   Δt³/6]
        [Δt⁴/8   Δt³/3   Δt²/2]
        [Δt³/6   Δt²/2   Δt   ]
```

**Hatása:**

| q | Mi történik | Amit a chartokon látsz |
|---|---|---|
| `1e-10` | Ultra-sima, alig mozdul | μ̂ szinte egyenes vonal, nem követi a valós mozgásokat |
| `1e-9` | Sima, lassú | A nagy trendeket követi, kis mozgásokat ignorálja |
| **`1e-8`** | **Alapértelmezett** | **Kompromisszum — ezt próbáld ki először** |
| `1e-7` | Gyorsabban reagál | Közelebb megy a nyers hozamokhoz, de még simít |
| `1e-6` | Szinte nincs simítás | A szűrő gyakorlatilag „kikapcsol" |

**Fizikai értelmezés:** A q azt modellezi, mennyire változékony a piac „igazi" gyorsulása (μ̈). Kriptón a volatilitás magas → relatíve nagy q kell (10⁻⁸ – 10⁻⁷). A `sensitivity_q.html` chart pontosan ezt mutatja — ott 5 különböző q-val fut a szűrő, és összehasonlíthatod.

**Hogyan hangold:** Ha az innováció variancia szisztematikusan nagyobb mint amit a szűrő vár (S_k) → a q túl kicsi, növeld. Ha kisebb → a q túl nagy.

---

### `sigma2_1m: null` — Az 1 perces log hozam varianciája

Ez az R (mérési zaj) mátrix alapegysége. Az összes TF mérési zaja ebből számolódik:

```
R_ij = min(n_i, n_j) · σ²_1m
```

Ha `null` → az adatból automatikusan becsüli: `Var(r_1m)` az összes 1 perces log hozamból. A 30 napos futásnál ez tipikusan **~10⁻⁷** nagyságrendű.

**Miért fontos:** Ez határozza meg, mennyire „zajos"-nak tekinti a szűrő a méréseket. Ha túl kicsi → a szűrő mindent elhisz (zajos output). Ha túl nagy → a szűrő figyelmen kívül hagyja a méréseket (túl sima). Az automatikus becslés általában jó kiindulópont.

---

### `h_mode: "discrete"` — A megfigyelési mátrix típusa

Két módszer van arra, hogyan kapcsoljuk a latens állapotot (μ, μ̇, μ̈) a megfigyelt hozamokhoz:

**`"continuous"`** — Folytonos integrálos közelítés:

```
H_τ = [τ,   ½τ²,   ⅙τ³]
```

Feltevés: μ(t) folytonos függvény, a hozam az integrálja. Egyszerű, elegáns, de közelítés.

**`"discrete"`** — Diszkrét aggregáció (egzakt):

```
H_n = [n,   -n(n-1)/2,   n(n-1)(2n-1)/12]
```

Feltevés: az n perces hozam pontosan n darab 1 perces hozam összege. Ez matematikailag egzakt.

**A különbség a gyakorlatban:** A `h_compare.html` chart pont ezt mutatja. Kis TF-eknél (1m, 5m) szinte nincs különbség. Nagy TF-eknél (4h, 1d) a diszkrét pontosabb, mert a folytonos közelítés Taylor-maradéktagja nő τ-val. A `"discrete"` a javasolt alapértelmezés.

---

### `r_mode: "full"` — A mérési zaj korrelációs struktúrája

**`"diagonal"`** — Egyszerűsített, független:

```
R = σ²_1m · diag(1, 5, 15, 60, 240, 1440)
```

Feltevés: a TF-ek mérési zajai függetlenek. **Ez hibás**, mert az 5m hozam tartalmazza az 1m hozamot!

**`"full"`** — Teljes kovariancia mátrix:

```
R_ij = min(n_i, n_j) · σ²_1m

R = σ²_1m · [  1    1    1    1     1     1   ]
             [  1    5    5    5     5     5   ]
             [  1    5   15   15    15    15   ]
             [  1    5   15   60    60    60   ]
             [  1    5   15   60   240   240   ]
             [  1    5   15   60   240  1440   ]
```

**Miért számít:** A `"full"` mód megmondja a szűrőnek, hogy a 15m hozam és az 5m hozam korreláltak (mert az 5m benne van a 15m-ben). Enélkül a szűrő **kétszer számolná** az átfedő információt → túlzott magabiztosság, alulbecsült P. A `"full"` a helyes választás.

---

### `P0_scale: 100.0` — Kezdeti bizonytalanság

```
P₀ = 100 · I₃ₓ₃
```

A szűrő indulásakor a P (kovariancia) mátrix. Azt fejezi ki: „fogalmam sincs, hol van a valódi állapot".

**Nagy P₀** (100) → a szűrő az első méréseket nagyon erősen figyelembe veszi (nagy K gain) → gyorsan konvergál a valós állapothoz. Ez a burn-in periódus, amit levágunk (első 50 lépés).

**Kis P₀** (0.01) → a szűrő „beképzelt", lassan hajlandó a mérésekhez igazodni → hosszabb konvergencia.

A 100-as érték bőven elég nagy, 2-3 lépés alatt a P leesik a stabil szintre.

---

## Trend paraméterek

### `w_mu: 0.50, w_mu_dot: 0.35, w_mu_ddot: 0.15` — Kompozit jel súlyok

A trend score:

```
score = 0.50·(μ̂/σ_μ) + 0.35·(μ̂̇/σ_μ̇) + 0.15·(μ̂̈/σ_μ̈)
```

Minden komponenst a saját szórásával normalizálunk (z-score), aztán súlyozzuk:

| Súly | Komponens | Mit mér | Miért ez a súly |
|------|-----------|---------|-----------------|
| **0.50** | μ̂ (szint) | „Hol tartunk?" — pozitív/negatív hozam ráta | A legmegbízhatóbb, legkevésbé zajos → legnagyobb súly |
| **0.35** | μ̂̇ (momentum) | „Merre megyünk?" — a trend iránya erősödik/gyengül | Erős jel, de μ̂-nél zajosabb → közepes súly |
| **0.15** | μ̂̈ (gyorsulás) | „Gyorsulunk?" — inflexiós pontok | A legzajosabb (harmadik derivált) → legkisebb súly |

A súlyok összege 1.0. Kísérletileg változtathatod: ha nagyobb súlyt adsz μ̈-nek → érzékenyebb a trendváltásokra, de zajosabb.

---

### `rolling_window: 120` — Normalizálási ablak (percben)

A trend score komponenseit z-score-ral normalizáljuk: `μ̂ / σ_μ`, ahol σ_μ az utolsó 120 perces (= 2 órás) ablak szórása.

**Hatása:**

- **Kis ablak** (30 perc) → az „átlagos" gyorsan változik, a score reaktívabb, több hamis jel
- **Nagy ablak** (480 perc = 8 óra) → stabilabb normalizálás, de lassabban adaptálódik a volatilitás-változásokhoz
- **120 perc** → 2 óra, jó kompromisszum: elég stabil normalizálás, de még követi a napi volatilitás-ciklusokat

**Miért kell normalizálni:** μ̂ ≈ 10⁻⁴, μ̂̈ ≈ 10⁻⁸ → nyers összeadásnál a μ̂ dominálna. A z-score azonos skálára hozza mindhárom komponenst.
