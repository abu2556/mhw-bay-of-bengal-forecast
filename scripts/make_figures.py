"""Manuscript Figs 2, 4, 6, 7 and supplementary Fig. S3, redrawn from the result tables in results/tables.
Run from the repository root:  python scripts/make_figures.py   ->  results/figures/manuscript/*.png (600 dpi)"""
import os, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

T = 'results/tables'; OUT = 'results/figures/manuscript'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'savefig.dpi': 600, 'savefig.bbox': 'tight'})
# one colour per entity across all figures (validated: lightness, chroma, CVD and normal-vision separation)
C = dict(causal='#4b5563', exceed='#1a9e77', logit='#2a76d2', rf='#e8601c', label='#b0479a',
         oisst='#2a76d2', cci='#e8601c', bob='#2a76d2', lev='#e8601c', scs='#1a9e77')
M = dict(causal='D', exceed='v', logit='o', rf='^', label=None)
LEADS = [1, 3, 5, 7, 10, 14]

def rng(s):
    a, b = re.findall(r'[+-−]?\d*\.\d+', str(s).replace('−', '-'))[:2]
    return float(a), float(b)

def style(ax, leads):
    ax.set_xticks(leads); ax.grid(True); ax.set_axisbelow(True); ax.set_xlabel('Lead (days)')

# ---------------- Fig. 2: schematic of the three benchmarks
def fig2():
    d = np.r_[-0.40, -0.30, -0.20, -0.10, 0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.45, 0.30, 0.20, 0.10,
              -0.15, -0.30, -0.20, -0.10, 0.10, 0.15, -0.10, -0.30, -0.35, -0.30, -0.40, -0.45]
    days = np.arange(1, 27)
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 3.6), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.15), sharex=True)
    a.axvspan(4.5, 8.5, color=C['label'], alpha=0.10, lw=0)
    a.text(6.5, 0.83, 'known only\nin hindsight', ha='center', va='top', color='#7a2f69', fontsize=7)
    a.bar(days, d, width=0.8, color=np.where(d > 0, C['rf'], '#a3aab4'), edgecolor='white', linewidth=0.5)
    a.axhline(0, color='#333333', lw=0.7)
    a.text(19.5, 0.42, '2-day exceedance:\nnever an MHW', ha='center', fontsize=7, color='#333333')
    a.set_ylabel('SST minus\nthreshold (°C)'); a.set_ylim(-0.52, 0.88); a.grid(True, axis='y'); a.set_axisbelow(True)
    rows = [('Official Hobday label', C['label'], list(range(5, 15))),
            ('Causal persistence state', C['causal'], list(range(9, 15))),
            ('Exceedance on the day', C['exceed'], list(range(5, 15)) + [19, 20])]
    for k, (name, col, on) in enumerate(rows):
        y = 2 - k
        b.bar(on, 0.62, bottom=y - 0.31, width=0.82, color=col, edgecolor='white', linewidth=0.6)
    b.set_yticks([2, 1, 0]); b.set_yticklabels([r[0] for r in rows]); b.set_ylim(-0.6, 2.6)
    b.tick_params(axis='y', length=0); b.spines['left'].set_visible(False)
    b.set_xticks([1, 5, 9, 14, 20, 26]); b.set_xlim(0.3, 26.7); b.set_xlabel('Day')
    fig.savefig(f'{OUT}/fig2_benchmark_schematic.png'); plt.close(fig)

# ---------------- Fig. 4 and Fig. S3: inflation curves
def fig4():
    d = pd.read_csv(f'{T}/wp1_leakage_curve_3regions.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    regs = [('Bay of Bengal', 'bob', 'o'), ('Levantine Sea', 'lev', 's'), ('South China Sea', 'scs', '^')]
    for ax, (tg, title) in zip(axs, (('det', '(a) Detrended labels'), ('raw', '(b) Raw labels'))):
        for reg, key, mk in regs:
            s = d[(d.region == reg) & (d.target == tg)].sort_values('lead')
            ax.fill_between(s.lead, s.inflation_total_lo, s.inflation_total_hi, color=C[key], alpha=0.13, lw=0)
            ax.plot(s.lead, s.inflation_total, color=C[key], marker=mk, ms=4, lw=1.4, label=reg)
        ax.axhline(0, color='#333333', lw=0.6); style(ax, [1, 2, 3, 5, 7, 10, 14]); ax.set_title(title)
    axs[0].set_ylabel('MCC inflation\n(label-based minus causal)')
    axs[0].legend(frameon=False, loc='upper right')
    fig.savefig(f'{OUT}/fig4_inflation_three_regions.png'); plt.close(fig)

def figS3():
    d = pd.read_csv(f'{T}/wp1c_grid_leakage_summary.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, (tg, title) in zip(axs, (('det', '(a) Detrended labels'), ('raw', '(b) Raw labels'))):
        s = d[d.labels == tg].sort_values('lead')
        lo, hi = zip(*[rng(v) for v in s['cell inflation 10th-90th pct']])
        ax.fill_between(s.lead, lo, hi, color=C['logit'], alpha=0.18, lw=0, label='Grid cells, 10th–90th percentile')
        ax.plot(s.lead, s['cell inflation median'], color=C['logit'], marker='o', ms=4, lw=1.4, label='Grid cells, median')
        ax.plot(s.lead, s['box-mean inflation'], color=C['causal'], marker='s', ms=4, lw=1.2, ls='--', label='Regional mean')
        ax.axhline(0, color='#333333', lw=0.6); style(ax, [1, 2, 3, 5, 7, 10, 14]); ax.set_title(title)
    axs[0].set_ylabel('MCC inflation\n(label-based minus causal)'); axs[1].legend(frameon=False, loc='upper right')
    fig.savefig(f'{OUT}/figS3_grid_vs_regional_inflation.png'); plt.close(fig)

# ---------------- Fig. 6: skill by lead
def fig6():
    sk = pd.read_csv(f'{T}/wp2_skill_by_lead_det.csv'); ex = pd.read_csv(f'{T}/wp2_exceedance_persistence_det.csv')
    lb = pd.read_csv(f'{T}/wp1_leakage_curve_bob.csv'); lb = lb[(lb.target == 'det') & lb.lead.isin(LEADS)].sort_values('lead')
    g = lambda m: sk[sk.model == m].sort_values('lead')
    ex = ex.sort_values('lead')
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.2), gridspec_kw=dict(wspace=0.42))
    series = [('Causal persistence', 'causal', g('persistence').MCC, g('persistence').MCC_lo, g('persistence').MCC_hi),
              ('Exceedance persistence', 'exceed', ex['exceedance persistence'], *zip(*[rng(v) for v in ex['exc. pers. 95% CI']])),
              ('Logistic regression, SST', 'logit', g('logit_sst').MCC, g('logit_sst').MCC_lo, g('logit_sst').MCC_hi),
              ('Random forest, SST + ERA5', 'rf', g('rf_era5').MCC, g('rf_era5').MCC_lo, g('rf_era5').MCC_hi)]
    for name, k, y, lo, hi in series:
        a.fill_between(LEADS, lo, hi, color=C[k], alpha=0.10, lw=0)
        a.plot(LEADS, y, color=C[k], marker=M[k], ms=4, lw=1.4, label=name)
    a.plot(LEADS, lb.MCC_label, color=C['label'], ls=':', lw=1.8, label='Label-based persistence')
    style(a, LEADS); a.set_ylabel('MCC'); a.set_title('(a) Skill by lead'); a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=1.8)
    lsst = g('logit_sst').MCC.values
    for k, base, cis, dx, mk, name in (('causal', g('persistence').MCC.values, g('logit_sst')['gain vs persistence'], -0.18, 'D', 'Over causal persistence'),
                                       ('exceed', ex['exceedance persistence'].values, ex['logit_sst gain vs exc. pers.'], 0.18, 'v', 'Over exceedance persistence')):
        lo, hi = map(np.array, zip(*[rng(v) for v in cis])); p = lsst - base
        b.errorbar(np.array(LEADS) + dx, p, yerr=[p - lo, hi - p], fmt=mk, ms=4.5, color=C[k], ecolor=C[k], elinewidth=1.1, capsize=0, label=name)
    b.axhline(0, color='#333333', lw=0.7); style(b, LEADS); b.set_ylabel('MCC gain of the logistic SST model\n(95% CI)')
    b.set_title('(b) Gain of the logistic SST model'); b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.savefig(f'{OUT}/fig6_skill_by_lead.png'); plt.close(fig)

# ---------------- Fig. 7: OISST vs CCI/C3S
def fig7():
    mo = pd.read_csv(f'{T}/v2_diag_month.csv', index_col=0); lk = pd.read_csv(f'{T}/v2_leakage_both.csv'); sk = pd.read_csv(f'{T}/v2_skill_both.csv')
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.8), gridspec_kw=dict(hspace=0.62, wspace=0.32)); (a, b), (c, d) = axs
    m = np.arange(1, 13)
    a.axvspan(5.5, 9.5, color='#eef0f3', lw=0, zorder=0)
    a.bar(m, mo.both / mo['MHW days OISST'], width=0.7, color='#a3aab4', label='OISST MHW days that are\nalso MHW days in CCI/C3S', zorder=2)
    a.plot(m, mo.r, color='#222222', marker='o', ms=3.5, lw=1.3, label='Correlation of daily anomalies', zorder=3)
    a.set_xticks(m); a.set_xticklabels(list('JFMAMJJASOND')); a.set_yticks(np.arange(0, 1.01, 0.2))
    a.set_xlabel('Calendar month'); a.set_ylabel('Fraction / correlation'); a.set_title('(a) Agreement between the products')
    a.text(7.5, 0.95, 'monsoon', ha='center', va='top', fontsize=6.5, color='#555555', zorder=4)
    a.set_ylim(0, 1.0); a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.5, handlelength=1.4); a.grid(True, axis='y'); a.set_axisbelow(True)
    for ds, key, ls, mk in (('OISST', 'oisst', '-', 'o'), ('CCI', 'cci', '--', 's')):
        s = lk[lk.dataset == ds].sort_values('lead')
        b.fill_between(s.lead, s.lo, s.hi, color=C[key], alpha=0.13, lw=0)
        b.plot(s.lead, s.inflation, color=C[key], ls=ls, marker=mk, ms=3.5, lw=1.4, label='OISST' if ds == 'OISST' else 'CCI/C3S')
    b.axhline(0, color='#333333', lw=0.6); style(b, [1, 2, 3, 5, 7, 10, 14]); b.set_ylabel('MCC inflation')
    b.set_title('(b) Label-based minus causal persistence'); b.legend(frameon=False, loc='upper right')
    for ds, ls in (('OISST', '-'), ('CCI', '--')):
        s = sk[sk.dataset == ds].sort_values('lead')
        for col, k in (('honest', 'causal'), ('exceedance', 'exceed'), ('logistic', 'logit')):
            c.plot(s.lead, s[col], color=C[k], ls=ls, marker=M[k], ms=3.5, lw=1.3)
    style(c, LEADS); c.set_ylabel('MCC'); c.set_title('(c) Skill: solid OISST, dashed CCI/C3S')
    c.legend(handles=[Line2D([], [], color=C['causal'], marker='D', ms=3.5, label='Causal persistence'),
                      Line2D([], [], color=C['exceed'], marker='v', ms=3.5, label='Exceedance persistence'),
                      Line2D([], [], color=C['logit'], marker='o', ms=3.5, label='Logistic, SST'),
                      Line2D([], [], color='#555555', ls='-', label='OISST'), Line2D([], [], color='#555555', ls='--', label='CCI/C3S')],
             frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.9, handlelength=1.8, fontsize=6.5)
    for ds, key, dx, mk in (('OISST', 'oisst', -0.2, 'o'), ('CCI', 'cci', 0.2, 's')):
        s = sk[sk.dataset == ds].sort_values('lead')
        lo, hi = map(np.array, zip(*[rng(v) for v in s['logistic vs honest']])); p = (s.logistic - s.honest).values
        d.errorbar(s.lead + dx, p, yerr=[p - lo, hi - p], fmt=mk, ms=4, color=C[key], ecolor=C[key], elinewidth=1.1, capsize=0,
                   label='OISST' if ds == 'OISST' else 'CCI/C3S')
    d.axhline(0, color='#333333', lw=0.7); style(d, LEADS); d.set_ylabel('MCC gain'); d.set_title('(d) Logistic SST minus causal persistence')
    d.legend(frameon=False, loc='upper right')
    fig.savefig(f'{OUT}/fig7_oisst_vs_cci.png'); plt.close(fig)

for f in (fig2, fig4, figS3, fig6, fig7): f()
print('written to', OUT)
