"""Past-only vs full-record reference, expanding-window forecasts 2002-2025 (detrended labels).
Rebuilt from data/processed/predictor_table_1982_2025.csv of the public repository."""
import sys, numpy as np, pandas as pd, warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

# Run from the repository root: python scripts/past_only_paired_test.py
import os
df = pd.read_csv('data/processed/predictor_table_1982_2025.csv', index_col=0, parse_dates=True)
OUT = 'results/past_only'; os.makedirs(OUT, exist_ok=True)
DATES = df.index
TESTY = list(range(2002, 2026))
LEADS = [1, 3, 5, 7, 10, 14]

def doy366(idx):
    d = idx.dayofyear.values.copy(); d[(~idx.is_leap_year) & (idx.month > 2)] += 1
    return d
D366 = doy366(DATES)

def hobday_clim(v, base, win=5, smooth=31, pct=90):
    yrs = DATES.year.values
    inb = (yrs >= base[0]) & (yrs <= base[1])
    m, p = np.full(367, np.nan), np.full(367, np.nan)
    for k in range(1, 367):
        if k == 60: continue
        idx = np.where((D366 == k) & inb)[0]
        pool = np.unique(np.clip((idx[:, None] + np.arange(-win, win + 1)).ravel(), 0, len(v) - 1))
        x = v[pool[inb[pool]]]
        m[k], p[k] = np.nanmean(x), np.nanpercentile(x, pct)
    out = []
    for c in (m, p):
        c[60] = (c[59] + c[61]) / 2; c = c[1:]
        c = np.convolve(np.r_[c[-(smooth // 2):], c, c[:smooth // 2]], np.ones(smooth) / smooth, 'valid')
        out.append(c[D366 - 1])
    return out

def detect(exc, min_dur=5, max_gap=2):
    e = np.asarray(exc, int); n = len(e)
    b = np.diff(np.r_[0, e, 0]); st, en = np.where(b == 1)[0], np.where(b == -1)[0] - 1
    keep = en - st + 1 >= min_dur; st, en = st[keep], en[keep]
    causal, label = np.zeros(n, int), np.zeros(n, int)
    for s0, e0 in zip(st, en): label[s0:e0 + 1] = 1; causal[s0 + min_dur - 1:e0 + 1] = 1
    for k in range(len(st) - 1):
        if st[k + 1] - en[k] - 1 <= max_gap: label[en[k] + 1:st[k + 1]] = 1
    return causal, label

def run_length(b):
    g = (~b).cumsum(); return b.astype(int).groupby(g).cumsum()

def features(sd, clim, thr):
    a = pd.Series(sd - clim, index=DATES); dist = pd.Series(sd - thr, index=DATES)
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
    return X

def reference(kind, Y):
    """SST series, climatology, threshold, labels for test year Y.
    kind: 'past'/'full' (detrended labels) or 'past_raw'/'full_raw' (raw labels)."""
    raw = df.sst.values.astype(float); t = np.arange(len(raw), dtype=float)
    if kind == 'full':
        sd = df.sst_detrended.values.astype(float); base = (1982, 2021)
    elif kind == 'full_raw':
        sd = raw; base = (1982, 2021)
    elif kind == 'past_raw':
        sd = raw; base = (1982, Y - 1)
    else:
        m = DATES.year.values <= Y - 1
        slope, icpt = np.polyfit(t[m], raw[m], 1)
        sd = raw - slope * t; base = (1982, Y - 1)
    clim, thr = hobday_clim(sd, base)
    causal, label = detect(sd > thr)
    return sd, clim, thr, causal, label

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

def fit_predict(tr, te, cols, inner_folds=4):
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
    return (p >= th).astype(int)

import sys
KINDS = [] if sys.argv[1:] == ['score'] else (sys.argv[1:] or ['past', 'full', 'past_raw', 'full_raw'])
for kind in KINDS:
  for h in LEADS:
      if os.path.exists(f'{OUT}/{kind}_h{h}.csv'):        # resume after a Colab disconnect
          print(kind, h, 'already done'); continue
      rows = []
      for Y in TESTY:
          sd, clim, thr, causal, label = reference(kind, Y)
          X = features(sd, clim, thr); cols = list(X.columns)
          lab = pd.Series(label, index=DATES)
          tb = X.assign(y=lab.shift(-h), year=DATES.year, target_year=(DATES + pd.Timedelta(days=h)).year,
                        target=DATES + pd.Timedelta(days=h), label_t=label).dropna()
          tb['y'] = tb.y.astype(int)
          te = tb[tb.year == Y]
          tr = tb[tb.target <= pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=10)]
          pred = fit_predict(tr, te, cols)
          rows.append(pd.DataFrame({'year': Y, 'y': te.y.values, 'model': pred, 'causal': te.mhw_causal.values,
                                    'exceed': te.exceed_now.values, 'label': te.label_t.values.astype(int),
                                    'label_day': lab.reindex(te.index).values}, index=te.index))
          print(kind, h, Y, flush=True)
      pd.concat(rows).to_csv(f'{OUT}/{kind}_h{h}.csv')

# ---------- scores, Table 6 and paired test (Table S13) ----------
B=2000; YRS=list(range(2002,2026)); IDX=np.random.default_rng(0).integers(0,len(YRS),(B,len(YRS)))
def mcc(tp,fp,fn,tn):
    tp,fp,fn,tn=(np.asarray(v,float) for v in (tp,fp,fn,tn)); den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1),0.0)
def counts(p,y,yr):
    g=pd.DataFrame({'tp':(p==1)&(y==1),'fp':(p==1)&(y==0),'fn':(p==0)&(y==1),'tn':(p==0)&(y==0),'yr':yr}).groupby('yr').sum()
    return g.reindex(YRS,fill_value=0).values
pt=lambda c: float(mcc(*c.sum(0))); bt=lambda c: mcc(*c[IDX].sum(1).T)
ci=lambda d: f'{np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f}'
rows=[]
for lt,(KP,KF) in {'detrended':('past','full'),'raw':('past_raw','full_raw')}.items():
  if not all(os.path.exists(f'{OUT}/{k}_h1.csv') for k in (KP,KF)): continue
  for h in (1,3,5,7,10,14):
      R={}
      for k,kk in (('past',KP),('full',KF)):
          o=pd.read_csv(f'{OUT}/{kk}_h{h}.csv',index_col=0,parse_dates=True)
          C={n:counts(o[n].values,o.y.values,o.year.values) for n in ('model','causal','exceed','label')}
          R[k]=(C,{n:bt(c) for n,c in C.items()},o)
      Cp,Bp,op=R['past']; Cf,Bf,of=R['full']
      gp=Bp['model']-Bp['exceed']; gf=Bf['model']-Bf['exceed']
      gpc=Bp['model']-Bp['causal']; gfc=Bf['model']-Bf['causal']
      rows.append({'labels':lt,'lead':h,'causal':round(pt(Cp['causal']),3),'exceed':round(pt(Cp['exceed']),3),'logit':round(pt(Cp['model']),3),
        'inflation':f"{pt(Cp['label'])-pt(Cp['causal']):+.3f} ({ci(Bp['label']-Bp['causal'])})",
        'gain_causal_past':f"{pt(Cp['model'])-pt(Cp['causal']):+.3f} ({ci(gpc)})",
        'gain_exc_past':f"{pt(Cp['model'])-pt(Cp['exceed']):+.3f} ({ci(gp)})",
        'gain_exc_full':f"{pt(Cf['model'])-pt(Cf['exceed']):+.3f} ({ci(gf)})",
        'DIFF_exc past-full':f"{(pt(Cp['model'])-pt(Cp['exceed']))-(pt(Cf['model'])-pt(Cf['exceed'])):+.3f} ({ci(gp-gf)})",
        'DIFF_causal past-full':f"{(pt(Cp['model'])-pt(Cp['causal']))-(pt(Cf['model'])-pt(Cf['causal'])):+.3f} ({ci(gpc-gfc)})"})
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/paired_test.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',20); print(T.to_string(index=False))
o1p=pd.read_csv(f'{OUT}/past_h1.csv',index_col=0,parse_dates=True); o1f=pd.read_csv(f'{OUT}/full_h1.csv',index_col=0,parse_dates=True)
print('MHW days 2002-2025 past-only:',int(o1p.label_day.sum()),'full:',int(o1f.label_day.sum()),
      'label MCC:',round(float(mcc(*counts(o1p.label_day.values,o1f.label_day.values,o1p.year.values).sum(0))),3))
