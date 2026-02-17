# PLAN — Multi-TF Kalman Filter Research & Visualization (BTC)

## Kontextus

A `KALMAN_LOG_MULTI_TF.md` dokumentum egy 3-állapotú Kalman-szűrőt ír le, ami BTC log-hozamokat szűr egyszerre több időkeretből (1m→1d). A cél: **a teljes elméletet Python-ban implementálni és minden aspektust interaktívan vizualizálni**. Nem kereskedési bot — tisztán kutatás.

---

## Projekt struktúra

```
multi_tf_kalman/
├── config.yaml                  # Minden paraméter egy helyen
├── config.py                    # Pydantic config modell + YAML betöltés
├── data/
│   ├── fetcher.py               # Binance OHLCV letöltés (ccxt)
│   └── cache/                   # Letöltött adat (parquet)
├── kalman/
│   ├── matrices.py              # F, H, Q, R mátrix építők
│   ├── filter.py                # MultiTFKalmanFilter osztály
│   └── smoother.py              # RTS simító
├── signals.py                   # Trend score, anomália, predikció
├── visualizations/
│   ├── base.py                  # BasePlot (plotly, export)
│   ├── viz_states.py            # Szűrt állapotok + ár
│   ├── viz_returns.py           # Nyers vs szűrt hozamok TF-enként
│   ├── viz_gain.py              # Kalman-nyereség dinamika
│   ├── viz_innovation.py        # Innováció + anomália detekció
│   ├── viz_covariance.py        # P mátrix evolúció
│   ├── viz_prediction.py        # Predikció vs valós
│   ├── viz_trend.py             # Trend score dashboard
│   ├── viz_sensitivity.py       # q paraméter érzékenység
│   ├── viz_h_compare.py         # Folytonos vs diszkrét H
│   └── viz_smoother.py          # Online vs RTS simított
├── run_research.py              # Fő futtatóscript (minden vizualizáció)
└── notebooks/
    └── explorer.ipynb           # Interaktív Jupyter notebook
```

---

## 1. Konfiguráció (`config.yaml` + `config.py`)

### `config.yaml`
```yaml
symbol: "BTC/USDT"
exchange: "binance"
timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
data:
  days_back: 7             # 7 nap 1m adat ≈ 10080 sor
  cache_dir: "data/cache"
kalman:
  q: 1e-8                  # gyorsulás spektrális sűrűsége
  sigma2_1m: null           # null = automatikus becslés adatból
  h_mode: "discrete"        # "continuous" | "discrete"
  r_mode: "full"            # "full" (nem-diagonális) | "diagonal"
  P0_scale: 100.0
trend:
  w_mu: 0.50
  w_mu_dot: 0.35
  w_mu_ddot: 0.15
  rolling_window: 120       # normalizáláshoz (percben)
visualization:
  format: "html"
  theme: "plotly_dark"
  width: 1920
  height: 1080
```

### `config.py`
- Pydantic `BaseModel` (mint a meglévő `kalman_fomo/config.py` minta)
- `TIMEFRAME_MAP`: `{"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}`
- Bővíthető: új TF hozzáadás = egyetlen sor a `timeframes` listába
- Validátor: timeframes rendezettség ellenőrzés (növekvő sorrend)
- `from_yaml()` class method

---

## 2. Adat pipeline (`data/fetcher.py`)

### Stratégia: Csak 1m gyertyákat töltjük le, a többi TF-et aggregáljuk

**Miért?** Így garantált a konzisztencia — az 5m hozam tényleg 5 db 1m hozam összege (egzakt aggregációs azonosság).

### Implementáció
- **ccxt** könyvtár (mint `kalman_benchmark/src/fetch_data.py`)
- Binance, `BTC/USDT`, `1m` timeframe
- Paginált letöltés (`limit=1000`, cursor-based)
- Parquet cache: `data/cache/BTCUSDT_1m_{start}_{end}.parquet`
- Ha a cache friss → nem tölt le újra

### Feldolgozás
```python
def compute_log_returns(df_1m: pd.DataFrame) -> dict[str, pd.Series]:
    """1m close-ból log hozamok minden TF-re."""
    log_price = np.log(df_1m["close"])
    returns = {}
    for tf_label, n_minutes in TIMEFRAME_MAP.items():
        returns[tf_label] = log_price - log_price.shift(n_minutes)
    return returns
```

- A `returns` dict kulcsai: `"1m"`, `"5m"`, ..., `"1d"`
- Minden Series azonos DatetimeIndex-szel (1m felbontás)
- NaN ahol az adott TF mérés nem elérhető (pl. 5m mérés csak 5 percenként)

---

## 3. Kalman mátrixok (`kalman/matrices.py`)

### Újrafelhasználandó minta
A meglévő `kalman_fomo/kalman/filter.py` struktúráját követjük (KalmanState dataclass, predict/update/step minta, P stabilizálás, history lista).

### F mátrix (Δt = 1 perc)
```python
def build_F(dt: float = 1.0) -> np.ndarray:
    return np.array([
        [1, dt, 0.5*dt**2],
        [0, 1,  dt        ],
        [0, 0,  1         ],
    ])
```

### H mátrix — két változat

```python
def build_H_continuous(tau: float) -> np.ndarray:
    """H_τ = [τ, ½τ², ⅙τ³]"""
    return np.array([[tau, 0.5*tau**2, tau**3/6.0]])

def build_H_discrete(n: int) -> np.ndarray:
    """H_n = [n, -n(n-1)/2, n(n-1)(2n-1)/12]"""
    return np.array([[n, -n*(n-1)/2.0, n*(n-1)*(2*n-1)/12.0]])
```

Teljes H megépítése az aktív TF-ekre:
```python
def build_H_matrix(active_tfs: list[int], mode: str) -> np.ndarray:
    builder = build_H_continuous if mode == "continuous" else build_H_discrete
    return np.vstack([builder(n) for n in active_tfs])
```

### R mátrix

```python
def build_R_full(active_tfs: list[int], sigma2_1m: float) -> np.ndarray:
    """Nem-diagonális R: R_ij = min(n_i, n_j) * σ²_1m"""
    k = len(active_tfs)
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            R[i, j] = min(active_tfs[i], active_tfs[j]) * sigma2_1m
    return R

def build_R_diagonal(active_tfs: list[int], sigma2_1m: float) -> np.ndarray:
    """Diagonális R: R_ii = n_i * σ²_1m"""
    return np.diag([n * sigma2_1m for n in active_tfs])
```

### Q mátrix
```python
def build_Q(q: float, dt: float = 1.0) -> np.ndarray:
    """Piecewise constant acceleration process noise."""
    return q * np.array([
        [dt**5/20, dt**4/8, dt**3/6],
        [dt**4/8,  dt**3/3, dt**2/2],
        [dt**3/6,  dt**2/2, dt     ],
    ])
```

---

## 4. Kalman-szűrő (`kalman/filter.py`)

### `MultiTFKalmanFilter` osztály

```python
@dataclass
class KalmanState:
    x: np.ndarray           # [3x1] állapot
    P: np.ndarray           # [3x3] kovariancia
    innovation: np.ndarray  # [kx1] innováció vektor (k = aktív TF-ek száma)
    S: np.ndarray           # [kxk] innováció kovariancia
    K: np.ndarray           # [3xk] Kalman gain
    mahalanobis: float      # χ² anomália metrika
    active_tfs: list[int]   # mely TF-ek voltak aktívak
```

**Fő logika — `step(k, measurements)`:**

1. **Predict:** `x̂ = F·x`, `P = F·P·Fᵀ + Q`
2. **Aktív TF-ek meghatározása:** `[n for n in all_tfs if k % n == 0]`
3. **Mérésvektor összeállítás:** `z = [returns[tf][k] for tf in active_tfs]`
4. **H és R megépítése:** csak az aktív TF sorok/almátrix
5. **Update:** standard Kalman korrekcó (multi-measurement)
6. **P stabilizálás:** szimmetrizálás + minimum sajátérték (meglévő minta)
7. **Innováció metrikák:** `d_k = ỹᵀ · S⁻¹ · ỹ`
8. **History mentés:** teljes KalmanState

**Kulcs eltérés a meglévő filter.py-tól:**
- A meglévő skalár z-t és 1D H-t kezel → az új **vektor z-t és változó méretű H/R-t** kezel
- `K = P @ H.T @ np.linalg.inv(S)` (mátrix inverzió, nem skalár osztás)
- Joseph-forma P update a numerikus stabilitásért: `P = (I-KH)P(I-KH)ᵀ + KRKᵀ`

---

## 5. RTS simító (`kalman/smoother.py`)

```python
def rts_smooth(states: list[KalmanState], F: np.ndarray) -> list[KalmanState]:
    """Rauch-Tung-Striebel backward pass."""
    # Visszafelé: k = N-2, N-3, ..., 0
    # C_k = P_k|k @ F.T @ inv(P_{k+1|k})
    # x̂_k|N = x̂_k|k + C_k @ (x̂_{k+1|N} - F @ x̂_k|k)
    # P_k|N = P_k|k + C_k @ (P_{k+1|N} - P_{k+1|k}) @ C_k.T
```

---

## 6. Jelzések (`signals.py`)

### Trend score
```python
def compute_trend_score(mu, mu_dot, mu_ddot, w, rolling_window):
    sigma_mu = mu.rolling(rolling_window).std()
    # ... normalizálás és súlyozás
    return w[0]*(mu/sigma_mu) + w[1]*(mu_dot/sigma_mu_dot) + w[2]*(mu_ddot/sigma_mu_ddot)
```

### Predikció
```python
def predict_return(x_hat, tau):
    """r̂ = μ̂·τ + μ̂̇·½τ² + μ̂̈·⅙τ³"""
    mu, mu_dot, mu_ddot = x_hat.flatten()
    return mu*tau + mu_dot*0.5*tau**2 + mu_ddot*tau**3/6.0
```

### Predikció konfidencia
```python
def prediction_confidence(P, tau, h_mode):
    H_tau = build_H(tau, h_mode)  # [1x3]
    var = (H_tau @ P @ H_tau.T).item()
    return 1.96 * np.sqrt(var)  # 95% CI félszélesség
```

---

## 7. Vizualizációk (10 db)

Minden vizualizáció: **Plotly interaktív HTML** (zoom, hover, pan). A meglévő `BasePlot` mintát követik (layout, export, price overlay).

### VIZ-1: Szűrt állapotok + ár (`viz_states.py`)
- **4 subplot egymás alatt**, közös x-tengely (idő)
- (1) BTC ár (gyertyagrafikonnal vagy vonallal)
- (2) μ̂ — szűrt pillanatnyi log hozam ráta (kék vonal, ±1σ sáv)
- (3) μ̂̇ — szűrt momentum (zöld/piros szín a pozitív/negatív tartománynak)
- (4) μ̂̈ — szűrt gyorsulás (hasonló szín)
- Háttérszín: halvány zöld ha μ̂>0, halvány piros ha μ̂<0

### VIZ-2: Nyers vs szűrt hozamok (`viz_returns.py`)
- **6 subplot** (egy minden TF-hez: 1m, 5m, 15m, 1h, 4h, 1d)
- Minden subplot: szürke = nyers hozam, kék = szűrt rekonstrukció (`H_τ · x̂`)
- Jobb felső sarokban: SNR javulás %-ban

### VIZ-3: Kalman gain dinamika (`viz_gain.py`)
- **Hőtérkép (heatmap)** időben: x = idő, y = gain mátrix elemek (K[i,j])
- Felső: a K mátrix 3 sora (μ, μ̇, μ̈ korrekció) TF-enként
- Vertikális jelölők: mikor frissülnek a nagy TF-ek (15m, 1h, 4h, 1d)
- Alsó panel: K norma (Frobenius) időben — ugrásokat mutat TF frissülésnél

### VIZ-4: Innováció + anomália (`viz_innovation.py`)
- **Felső panel:** normalizált innováció TF-enként (külön szín)
- **Alsó panel:** Mahalanobis-távolság (d_k) időben + χ² küszöb (95%, 99%)
- Anomália pontok kiemelve (piros kör)
- Tooltip: melyik TF-en volt a legnagyobb meglepetés

### VIZ-5: P mátrix evolúció (`viz_covariance.py`)
- **3 vonal:** P[0,0] (μ bizonytalanság), P[1,1] (μ̇), P[2,2] (μ̈)
- Log skála (y-tengely)
- Vertikális vonalak: nagy TF frissítéseknél a P lecsökken → "fűrészfog" minta
- Összesített konfidencia: `1/tr(P)`

### VIZ-6: Predikció pontosság (`viz_prediction.py`)
- **3 horizont:** 5m, 15m, 60m előrejelzés
- Scatter plot: prediktált vs tényleges hozam (ideális = 45°-os egyenes)
- Időbeli panel: predikció ± 95% CI sáv + tényleges hozam
- RMSE, MAE, hit rate (előjel-egyezés %) kiírva

### VIZ-7: Trend score dashboard (`viz_trend.py`)
- **Felső:** BTC ár, háttérben trend_score szerinti színezés (gradiens: piros→zöld)
- **Alsó:** 3 komponens halmozott area chart (μ̂, μ̂̇, μ̂̈ normalizált hozzájárulás)
- Jelmagyarázat a kompozit jel értelmezéséhez

### VIZ-8: Paraméter érzékenység (`viz_sensitivity.py`)
- **q paraméter sweep:** 5-7 különböző q érték (log skálán)
- Overlay: szűrt μ̂ minden q-ra (halvány színek), ár a háttérben
- Alsó panel: innovációs variancia-arány (empirikus/elméleti) → megmutatja melyik q a legjobb
- Ezt nem real-time futtatjuk, hanem a teljes adaton batch-ben

### VIZ-9: H mátrix összehasonlítás (`viz_h_compare.py`)
- **Felső:** μ̂ a continuous H-val vs discrete H-val (két vonal)
- **Középső:** eltérés a kettő között
- **Alsó:** innováció variancia mindkét módban
- Kérdés amit megválaszol: van-e érdemi különbség a gyakorlatban?

### VIZ-10: RTS simító vs online (`viz_smoother.py`)
- **Felső:** μ̂ online (szűrt) vs μ̂ simított (RTS) — utóbbi simább
- **Középső:** μ̂̇ online vs simított
- **Alsó:** P[0,0] online vs simított — simított alacsonyabb
- Kiemelés: inflexiós pontok, ahol a simított jel korábban jelez

---

## 8. Futtató script (`run_research.py`)

```python
def main():
    config = Config.from_yaml("config.yaml")

    # 1. Adat letöltés / cache betöltés
    df_1m = fetch_or_load(config)

    # 2. Log hozamok kiszámítása minden TF-re
    returns = compute_log_returns(df_1m)

    # 3. σ²_1m automatikus becslés (ha config-ban null)
    sigma2_1m = estimate_sigma2(returns["1m"])

    # 4. Kalman szűrő futtatás
    kf = MultiTFKalmanFilter(config)
    kf.run(returns)

    # 5. RTS simítás
    smoothed = rts_smooth(kf.history, kf.F)

    # 6. Jelzések számítása
    signals = compute_signals(kf, config)

    # 7. Vizualizációk generálása (mind a 10)
    generate_all_visualizations(config, df_1m, kf, smoothed, signals)
```

---

## 9. Függőségek

```
numpy
pandas
plotly
ccxt
pydantic
pyyaml
pyarrow         # parquet I/O
kaleido         # opcionális PNG export
```

---

## 10. Implementációs sorrend

| # | Modul | Leírás | Függ |
|---|-------|--------|------|
| 1 | `config.yaml` + `config.py` | Konfiguráció | — |
| 2 | `data/fetcher.py` | Adat letöltés + cache + log return számítás | 1 |
| 3 | `kalman/matrices.py` | F, H, Q, R mátrix építők | 1 |
| 4 | `kalman/filter.py` | MultiTFKalmanFilter | 1, 3 |
| 5 | `kalman/smoother.py` | RTS simító | 4 |
| 6 | `signals.py` | Trend score, predikció, anomália | 4 |
| 7 | `visualizations/base.py` | BasePlot | 1 |
| 8 | `visualizations/viz_*.py` | Mind a 10 vizualizáció | 4-7 |
| 9 | `run_research.py` | Futtató | mind |
| 10 | Smoke test | Futtatás 1 nap adaton, vizualizációk ellenőrzése | 9 |

---

## 11. Verifikáció

1. **Smoke test:** `python run_research.py` — lefut hiba nélkül, 10 HTML generálódik
2. **Sanity check:** q=0 → μ̂ konstans (nincs frissítés); q=∞ → μ̂ ≈ nyers 1m hozam
3. **Aggregációs konzisztencia:** `H_5m @ x̂ ≈ returns["5m"]` (szűrt rekonstrukció közel a mérthez)
4. **P evolúció:** nagy TF frissülésnél P csökken (fűrészfog minta látható VIZ-5-ben)
5. **Innováció whiteness:** normalizált innováció autokorrelációja ≈ 0 (ha a szűrő jól van hangolva)
6. **H összehasonlítás:** VIZ-9 megmutatja a continuous vs discrete különbséget
7. **RTS vs online:** simított becslés P-je ≤ online P (VIZ-10-ben ellenőrizhető)

---

## 12. Bővíthetőség

- **Új TF hozzáadása:** `config.yaml` → `timeframes` listába beírni (pl. `"30m"`)
- **Új coin:** `config.yaml` → `symbol` módosítás
- **Volatilitás állapot:** állapotvektor bővítés `[μ, μ̇, μ̈, σ², σ̇²]` — F, H, Q méret nő
- **Multi-coin:** minden coinra külön szűrő, majd cross-coin vizualizáció
