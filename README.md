# Multi-TF Kalman Filter (BTC)

Kutatási célú implementáció egy **multi-timeframe Kalman-szűrőhöz** BTC log-hozamokra, interaktív Plotly vizualizációkkal és RTS simítással.

> Ez a repository kutatásra és elemzésre készült, **nem** élő kereskedő bot.

## Mit csinál a projekt?

- Letölti (vagy cache-ből betölti) a Binance `BTC/USDT` **1 perces OHLCV** adatait (`ccxt`)
- Azonos bázis idősorból építi fel a több idősíkú log-hozamokat a konzisztencia miatt
- Futtat egy 3 állapotú Kalman-szűrőt a latens dinamikára:
  - `mu` (szint / szűrt hozamráta)
  - `mu_dot` (momentum)
  - `mu_ddot` (gyorsulás)
- Támogatja:
  - folytonos vs diszkrét megfigyelési mátrixot (`h_mode`)
  - teljes vs diagonális mérési kovarianciát (`r_mode`)
- Kiszámolja a kutatási jelzéseket:
  - trend score
  - prediktált hozam + konfidencia intervallum
  - innováció-alapú anomália jelölések
- **10 darab interaktív HTML dashboardot** generál az `output/` mappába

---

## Projektstruktúra

```text
multi_tf_kalman/
├── config.yaml
├── config.py
├── run_research.py
├── signals.py
├── data/
│   ├── fetcher.py
│   └── cache/
├── kalman/
│   ├── matrices.py
│   ├── filter.py
│   └── smoother.py
├── visualizations/
│   ├── base.py
│   ├── viz_states.py
│   ├── viz_returns.py
│   ├── viz_gain.py
│   ├── viz_innovation.py
│   ├── viz_covariance.py
│   ├── viz_prediction.py
│   ├── viz_trend.py
│   ├── viz_sensitivity.py
│   ├── viz_h_compare.py
│   └── viz_smoother.py
├── output/
└── 1 - KF_LOG_RETURN_MULTI_TF.md
```

---

## Telepítés

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install numpy pandas plotly ccxt pydantic pyyaml pyarrow scipy kaleido
```

---

## Gyors indítás

```bash
python run_research.py
```

Opcionális felülírások:

```bash
python run_research.py --config config.yaml
python run_research.py --days 3
python run_research.py --q 1e-8
```

Az output fájlok alapértelmezetten az `output/` mappába kerülnek (`config.yaml` alapján).

---

## Konfiguráció

A fő futási paraméterek a `config.yaml` fájlban vannak:

- `symbol`, `exchange`
- `timeframes` (jelenlegi alapérték: `["1m", "5m", "15m", "30m", "1h"]`)
- `data.days_back`, `data.cache_dir`
- `kalman.q`, `kalman.sigma2_1m`, `kalman.h_mode`, `kalman.r_mode`, `kalman.P0_scale`
- `trend.w_mu`, `trend.w_mu_dot`, `trend.w_mu_ddot`, `trend.rolling_window`
- `visualization.format`, `visualization.theme`, `visualization.output_dir`

---

## Generált vizualizációk (10 db)

- `filtered_states.html` - szűrt állapotok + ár
- `returns_comparison.html` - nyers vs rekonstruált hozamok timeframe-enként
- `kalman_gain_dynamics.html` - Kalman-nyereség dinamika
- `innovation_anomaly.html` - innováció és anomália detekció
- `covariance_evolution.html` - kovariancia evolúció (`P00`, `P11`, `P22`)
- `prediction_accuracy.html` - predikciós pontosság (5m/15m/60m)
- `trend_dashboard.html` - kompozit trend score dashboard
- `sensitivity_q.html` - q paraméter érzékenység
- `h_compare.html` - continuous vs discrete H összehasonlítás
- `smoother_rts.html` - online szűrés vs RTS simítás

---

## Dokumentáció

Részletes projektleírás (magyarul):

- [Elmélet: Kalman multi-TF log-hozamokon](./1%20-%20KF_LOG_RETURN_MULTI_TF.md)
- [Implementációs terv](./2%20-%20PLAN.md)
- [Paraméter útmutató](./3%20-%20PARAMS.md)
- [Vizualizációs értelmezési útmutató](./4%20-%20VIZ.md)

---

## Megjegyzések

- A pipeline kutatási reprodukálhatóságra és chart-alapú diagnosztikára van optimalizálva.
- A cache-elt parquet fájlok a `data/cache/` mappába kerülnek.
- A projekt **nem** minősül befektetési tanácsadásnak.

---

## Jogi nyilatkozat

Ez a repository kizárólag oktatási és kutatási célokat szolgál.
A kriptopiac rendkívül volatilis; minden használat saját felelősségre történik.
