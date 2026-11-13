"""Previous-winter ENSO test with the Relative Oceanic Nino Index (RONI, ERSSTv6, NOAA CPC) in place of the ONI.
roni_ndj.csv: NDJ RONI by the year of November (from the CPC RONI table, accessed November 2026)."""
import os, numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv('data/processed/predictor_table_1982_2025.csv', index_col=0, parse_dates=True)
r = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roni_ndj.csv')).set_index('year').roni_ndj
YRS = np.arange(1983, 2026)
X = pd.Series(r.reindex(YRS - 1).values, index=YRS)
out = {}
for lt in ('mhw_detrended', 'mhw_raw'):
    lab = df[lt].astype(int); onset = (lab == 1) & (lab.shift(1, fill_value=0) == 0)
    for name, months in (('FebDec', range(2, 13)), ('MAMJ', range(3, 7))):
        m = lab.index.month.isin(list(months))
        days = lab[m].groupby(lab.index[m].year).sum().reindex(YRS, fill_value=0)
        ev = onset[m].groupby(lab.index[m].year).sum().reindex(YRS, fill_value=0)
        for resp, y in (('days', days), ('events', ev)):
            rho, p = stats.spearmanr(X, y)
            loo = [stats.spearmanr(np.delete(X.values, i), np.delete(y.values, i))[0] for i in range(len(YRS))]
            out[(lt, f'{resp}_{name}')] = (round(rho, 2), round(p, 3), f'{min(loo):+.2f} to {max(loo):+.2f}', round(stats.pearsonr(X, y)[0], 2))
T = pd.DataFrame(out, index=['Spearman', 'p', 'LOYO range', 'Pearson']).T
print(T.to_string())
lab = df.mhw_detrended.astype(int); m = lab.index.month >= 2
days = lab[m].groupby(lab.index[m].year).sum().reindex(YRS, fill_value=0)
ph = pd.cut(X, [-np.inf, -0.5, 0.5, np.inf], right=False, labels=['La Nina', 'Neutral', 'El Nino'])
print(days.groupby(ph, observed=True).agg(['mean', 'count']).round(1))
el = ph == 'El Nino'
print('Mann-Whitney one-sided p =', round(stats.mannwhitneyu(days[el], days[~el], alternative='greater').pvalue, 3))
top = days.nlargest(5)
print('top-5 years:', {int(y): (int(d), float(X[y]), str(ph[y])) for y, d in top.items()})
