# Results

- `tables/` – the result tables behind Chapter 4 of the thesis, exactly as written by the notebooks. The main README lists which table or figure of the thesis each file belongs to.
- `figures/` – the main figures, as produced by the notebooks.

`tables/wp1c_grid_leakage_cells.csv` has one row per grid cell (805 cells) with its latitude and longitude, and for each label type (`det`, `raw`) and lead: the inflation (`infl`), its 95% CI (`lo`, `hi`) and exceedance minus honest persistence (`gap_exc`), plus the share of MHW days known only in hindsight (`hind_det`, `hind_raw`).

Abbreviations: MCC = Matthews correlation coefficient; BSS = Brier skill score; LOYO = leave-one-year-out; det = detrended labels; raw = raw-SST labels.
