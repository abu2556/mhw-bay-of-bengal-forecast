"""Label used as a model input: logistic SST model with and without the official Hobday label of the
forecast day as an extra predictor (detrended labels, OISST, leave-one-year-out, 1982-2025).
Run from the repository root:  python scripts/label_input_test.py [lead ...]"""
import os, sys, numpy as np, pandas as pd, warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/predictor_table_1982_2025.csv', index_col=0, parse_dates=True)
OUT = 'results/label_input'; os.makedirs(OUT, exist_ok=True)
DATES = df.index

def doy366(idx):
    d = idx.dayofyear.values.copy(); d[(~idx.is_leap_year) & (idx.month > 2)] += 1
    return d

def hobday_mean(s, base=(1982, 2021), win=5, smooth=31):
    v, d = s.values, doy366(s.index)
    inb = (s.index.year >= base[0]) & (s.index.year <= base[1])
    c = np.full(367, np.nan)
    for k in range(1, 367):
        if k == 60: continue
        idx = np.where((d == k) & inb)[0]
        pool = np.unique(np.clip((idx[:, None] + np.arange(-win, win + 1)).ravel(), 0, len(v) - 1))
        c[k] = np.nanmean(v[pool[inb[pool]]])
    c[60] = (c[59] + c[61]) / 2; c = c[1:]
    c = np.convolve(np.r_[c[-(smooth // 2):], c, c[:smooth // 2]], np.ones(smooth) / smooth, 'valid')
    return pd.Series(c[d - 1], index=s.index)

def run_length(b):
    g = (~b).cumsum(); return b.astype(int).groupby(g).cumsum()

sst = df.sst_detrended; clim = hobday_mean(sst); thr = df.thresh90_detrended
a, dist = sst - clim, sst - thr
X = pd.DataFrame(index=DATES)
X['sst_anom'], X['sst_dist_thr'] = a, dist
for k in (1, 3, 7): X[f'lag{k}'] = a.shift(k)
for w in (3, 7, 14, 30): X[f'rm{w}'] = a.rolling(w).mean()
X['sd7'], X['max7'] = a.rolling(7).std(), a.rolling(7).max()
for k in (1, 3, 7): X[f'chg{k}'] = a - a.shift(k)
e = dist > 0
X['exceed_now'] = e.astype(int); X['exceed_run'] = run_length(e); X['mhw_causal'] = (X.exceed_run >= 5).astype(int)
doy = DATES.dayofyear
X['doy_sin'], X['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
SST_COLS = list(X.columns)
X['label_t'] = df.mhw_detrended.astype(int)          # official Hobday label of the forecast day (uses later days)

def mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(v, float) for v in (tp, fp, fn, tn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)

def best_threshold(y, p):
    u = np.unique(p)
    extra = u if len(u) <= 500 else np.quantile(p, np.linspace(0.5, 1, 301))
    th = np.unique(np.r_[np.arange(0.01, 1.0, 0.01), extra]); th = th[(th > 0) & (th < 1)]
    P = p[None, :] >= th[:, None]
    m = mcc((P & (y == 1)).sum(1), (P & (y == 0)).sum(1), ((~P) & (y == 1)).sum(1), ((~P) & (y == 0)).sum(1))
    return th[np.argmax(m)], m.max()

def train_mask(tab, test_years):
    m = ~tab.year.isin(test_years) & ~tab.target_year.isin(test_years)
    after = tab.year.isin([y + 1 for y in test_years]) & (tab.index.dayofyear <= 30)
    return m & ~after

logit = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000))

def loyo(tab, cols, inner_folds=4):
    out = []
    for y in sorted(tab.year.unique()):
        tr, te = tab[train_mask(tab, [y])], tab[tab.year == y]
        chunks = np.array_split(np.array(sorted(tr.year.unique())), inner_folds)
        best = None
        for C in (0.01, 0.1, 1.0):
            oof = pd.Series(np.nan, index=tr.index)
            for ch in chunks:
                itr, iva = tr[train_mask(tr, list(ch))], tr[tr.year.isin(ch)]
                oof.loc[iva.index] = logit(C).fit(itr[cols], itr.y).predict_proba(iva[cols])[:, 1]
            ok = oof.notna(); sc = best_threshold(tr.y[ok].values, oof[ok].values)[1]
            if best is None or sc > best[0]: best = (sc, C, oof[ok], tr.y[ok])
        _, C, oof, yo = best
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof.values, yo.values)
        th, _ = best_threshold(yo.values, iso.predict(oof.values))
        p = iso.predict(logit(C).fit(tr[cols], tr.y).predict_proba(te[cols])[:, 1])
        out.append(pd.Series((p >= th).astype(int), index=te.index))
    return pd.concat(out)

LEADS = [int(v) for v in sys.argv[1:] if v != 'score'] or ([] if sys.argv[1:] == ['score'] else [1, 3, 5, 7, 10, 14])
lab = df.mhw_detrended
for h in LEADS:
    if os.path.exists(f'{OUT}/lead{h}.csv'):                # resume after a Colab disconnect
        print('lead', h, 'already done'); continue
    tb = X.assign(y=lab.shift(-h), year=DATES.year, target_year=(DATES + pd.Timedelta(days=h)).year).dropna()
    tb['y'] = tb.y.astype(int)
    pd.DataFrame({'year': tb.year, 'y': tb.y, 'causal': tb.mhw_causal, 'exceed': tb.exceed_now, 'label': tb.label_t,
                  'sst': loyo(tb, SST_COLS), 'sst_label': loyo(tb, SST_COLS + ['label_t'])}).to_csv(f'{OUT}/lead{h}.csv')
    print('lead', h, 'done', flush=True)

# ---------- scores
YRS = list(range(1982, 2026)); IDX = np.random.default_rng(0).integers(0, len(YRS), (2000, len(YRS)))
def counts(p, y, yr):
    g = pd.DataFrame({'tp': (p == 1) & (y == 1), 'fp': (p == 1) & (y == 0), 'fn': (p == 0) & (y == 1), 'tn': (p == 0) & (y == 0), 'yr': yr}).groupby('yr').sum()
    return g.reindex(YRS, fill_value=0).values
pt = lambda c: float(mcc(*c.sum(0))); bt = lambda c: mcc(*c[IDX].sum(1).T)
ci = lambda d: f'{np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}'
rows = []
for h in (1, 3, 5, 7, 10, 14):
    f = f'{OUT}/lead{h}.csv'
    if not os.path.exists(f): continue
    o = pd.read_csv(f, index_col=0, parse_dates=True)
    C = {n: counts(o[n].values, o.y.values, o.year.values) for n in ('causal', 'exceed', 'label', 'sst', 'sst_label')}
    Bt = {n: bt(c) for n, c in C.items()}
    rows.append({'lead': h, 'MCC SST model': round(pt(C['sst']), 3), 'MCC SST + label': round(pt(C['sst_label']), 3),
                 'gain from label input': f"{pt(C['sst_label']) - pt(C['sst']):+.3f} ({ci(Bt['sst_label'] - Bt['sst'])})",
                 'label-based persistence': round(pt(C['label']), 3),
                 'SST + label minus label-based persistence': f"{pt(C['sst_label']) - pt(C['label']):+.3f} ({ci(Bt['sst_label'] - Bt['label'])})",
                 'SST + label minus causal persistence': f"{pt(C['sst_label']) - pt(C['causal']):+.3f} ({ci(Bt['sst_label'] - Bt['causal'])})"})
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/label_input_test.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 20); print(T.to_string(index=False))
