"""Supplementary Fig. S2: forecasts for the three strongest detrended MHWs at 7- and 14-day leads.
Same event rule and data as notebook cell WP5-2 (largest cumulative intensity, 1982-2025); only the
figure is changed: 'Causal' spelling, readable date ticks, and the colours used in the other figures.
Usage:  python scripts/figS2_case_studies.py <PROC folder on Drive> <output folder>"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates

PROC, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5,
                     'savefig.dpi': 600, 'savefig.bbox': 'tight'})
C = dict(causal='#4b5563', exceed='#1a9e77', logit='#2a76d2', obs='#e8601c')   # same colours as Figs 2, 4, 6, 7

rd = lambda f: pd.read_csv(f'{PROC}/{f}', index_col=0, parse_dates=True)
pt = rd('predictor_table_1982_2025.csv')
lab = pt.mhw_detrended.astype(int)
exc = (pt.sst_detrended > pt.thresh90_detrended).astype(int)
anom = rd('features_det_t+7.csv').sst_anom.reindex(lab.index)
# lead 7 = Phase 5 runs (oof_p5), lead 14 = WP2 runs (oof_lead14), exactly as in WP2-1
SRC = {7: 'oof_p5', 14: 'oof_lead14'}
RES = {h: {n: rd(f'{d}/{n}_det.csv') for n in ('logit_sst', 'persistence')} for h, d in SRC.items()}

# 1) events; selection rule fixed before looking at forecasts: largest cumulative intensity
b = lab.values; d = np.diff(np.r_[0, b, 0])
st, en = np.where(d == 1)[0], np.where(d == -1)[0] - 1
EV = pd.DataFrame({'onset': lab.index[st], 'end': lab.index[en], 'duration_d': en - st + 1,
                   'cum_intensity_Cdays': [anom.iloc[s:e + 1].sum() for s, e in zip(st, en)]})
TOP = EV.sort_values('cum_intensity_Cdays', ascending=False).head(3).sort_values('onset').reset_index(drop=True)
print(TOP.assign(cum_intensity_Cdays=TOP.cum_intensity_Cdays.round(1)).to_string(), '\n')

# 2) forecasts placed on their target date (issue date + lead)
def at_target(s, h):
    s = s.copy(); s.index = s.index + pd.Timedelta(days=h); return s
FC = {}
for h in SRC:
    m = RES[h]['logit_sst']
    FC[h] = {'prob': at_target(m.prob, h), 'thr': at_target(m.thr, h), 'logistic (SST)': at_target(m.pred, h),
             'exceedance persistence': at_target(exc.reindex(m.index), h),
             'causal persistence': at_target(RES[h]['persistence'].pred, h)}

# 3) per-event scores (Table S7)
day = pd.Timedelta(days=1); rows = []
for _, ev in TOP.iterrows():
    evd = pd.date_range(ev.onset, ev.end); pre = pd.date_range(ev.onset - 30 * day, ev.onset - day)
    quiet_pre = pre[lab.reindex(pre).fillna(1).values == 0]
    for h in SRC:
        for n in ('logistic (SST)', 'exceedance persistence', 'causal persistence'):
            s_ev = FC[h][n].reindex(evd).fillna(0); first = s_ev[s_ev == 1].index.min()
            rows.append({'event onset': f'{ev.onset:%Y-%m-%d}', 'lead': h, 'forecast': n,
                         'first event day caught (days after onset)': (first - ev.onset).days if pd.notna(first) else 'never',
                         'event days forecast (%)': int(round(100 * s_ev.mean())),
                         'false alarms, 30 d before onset': int(FC[h][n].reindex(quiet_pre).fillna(0).sum())})
CS = pd.DataFrame(rows); CS.to_csv(f'{OUT}/wp5_case_studies.csv', index=False)
print(CS.to_string(index=False))

# 4) figure: 3 events x 2 leads
fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.4), sharey=True)
for i, ev in TOP.iterrows():
    win = pd.date_range(ev.onset - 30 * day, ev.end + 15 * day)
    obs = lab.reindex(win).fillna(0).values
    for j, h in enumerate(SRC):
        ax = axs[i, j]
        ax.fill_between(win, 0, 1, where=obs == 1, color=C['obs'], alpha=0.18, lw=0, step='mid', label='Observed MHW')
        ax.plot(win, FC[h]['prob'].reindex(win), color=C['logit'], lw=1.6, label='Logistic SST model, probability')
        ax.step(win, FC[h]['thr'].reindex(win), where='mid', color=C['logit'], ls='--', lw=0.9, label='Decision threshold')
        for k, (n, c, lb) in enumerate((('exceedance persistence', C['exceed'], 'Exceedance persistence: yes'),
                                        ('causal persistence', C['causal'], 'Causal persistence: yes'))):
            yes = (FC[h][n].reindex(win).fillna(0) == 1).values
            ax.plot(win[yes], np.full(yes.sum(), -0.07 - 0.07 * k), '|', color=c, ms=6, mew=1.2, label=lb)
        ax.set_ylim(-0.2, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(win[0], win[-1])
        ax.set_title(f'({"abcdef"[2 * i + j]}) {ev.onset:%b %Y} event, {h}-day lead', loc='left')
        # one labelled tick on the 1st of each month, unlabelled ticks on the 15th: no overlapping dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('1 %b\n%Y'))
        ax.grid(True, which='major'); ax.grid(True, which='minor', color='#f0f0f0', lw=0.4)
    axs[i, 0].set_ylabel('Forecast probability')
hh, ll = axs[0, 0].get_legend_handles_labels()
fig.legend(hh, ll, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.045))
fig.tight_layout(h_pad=1.2)
fig.savefig(f'{OUT}/figS2_case_studies.png'); print('\nwritten', f'{OUT}/figS2_case_studies.png')
