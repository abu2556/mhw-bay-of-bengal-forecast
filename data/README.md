# Processed data

## `processed/predictor_table_1982_2025.csv`

One row per day, 1 January 1982 – 31 December 2025 (16,071 days), no missing values. All series are area-weighted (cosine of latitude) means over the 805 ocean cells of the study box, 15–23°N, 85–93°E, unless stated otherwise. Built by notebooks 01 and 02.

| Column | Unit | Description |
|---|---|---|
| (index) | date | day |
| `t2m` | °C | ERA5 2 m air temperature, daily mean |
| `u10`, `v10` | m s⁻¹ | ERA5 10 m zonal and meridional wind |
| `wspd` | m s⁻¹ | daily mean of the hourly 10 m wind speed |
| `swnet`, `lwnet` | W m⁻² | ERA5 net shortwave and net longwave radiation at the surface |
| `lhf`, `shf` | W m⁻² | ERA5 latent and sensible heat flux |
| `qnet` | W m⁻² | net surface heat flux = swnet + lwnet + lhf + shf (ERA5 convention: positive downward, into the ocean) |
| `dmi` | °C | Dipole Mode Index from OISST (western minus south-eastern tropical Indian Ocean SST anomaly, 1982–2021 climatology) |
| `iobw_dt` | °C | Indian Ocean basin-wide SST anomaly, 40–100°E, 20°S–20°N, from OISST, linearly detrended |
| `nino34` | °C | NOAA CPC monthly Niño 3.4 anomaly (ERSST v5, 1991–2020 base); each month's value is assigned to every day of the **following** month, so it was available on that day |
| `rmm1`, `rmm2` | – | Real-time Multivariate MJO index (Australian Bureau of Meteorology) |
| `sst` | °C | regional-mean NOAA OISST v2.1 SST |
| `clim_mean`, `thresh90` | °C | Hobday et al. (2016) climatological mean and 90th-percentile threshold, 1982–2021 baseline |
| `ssta` | °C | sst − clim_mean |
| `mhw_raw` | 0/1 | Hobday MHW label on raw SST (1652 MHW days, 133 events) |
| `sst_detrended` | °C | linearly detrended SST (trend 0.211 °C per decade removed) |
| `thresh90_detrended` | °C | 90th-percentile threshold of the detrended series |
| `mhw_detrended` | 0/1 | Hobday MHW label on detrended SST, the main forecast target (1192 MHW days, 109 events) |

The MHW labels use the full Hobday definition (at least five days above the threshold, gaps of two days or less joined), so the label of a day depends on the following days. Do not use them as a persistence forecast; see the thesis, Section 3.4.

## `processed/godas_box_pentad.csv`

NCEP GODAS pentad fields averaged over the 147 cells of the study box that are ocean down to 45 m, 1981–2024 (notebook 11).

| Column | Unit | Description |
|---|---|---|
| `time` | date | pentad date |
| `t50` | °C | mean potential temperature of the 5, 15, 25, 35 and 45 m levels |
| `cells_t50` | – | number of cells in the mean |
| `ild` | m | isothermal-layer depth |
| `cells_ild` | – | number of cells with a defined isothermal-layer depth |

The pentad of 27 September 2024 could not be downloaded and was filled by linear interpolation (its cell counts are empty).

## Not included

Raw OISST, ERA5, CCI/C3S and GODAS files, and the large intermediate files (feature tables, out-of-fold forecasts), are not included because of their size. The notebooks download or rebuild all of them.

## Sources and acknowledgement

NOAA OISST v2.1 (NOAA PSL); ERA5 (Copernicus Climate Change Service, ECMWF) – contains modified Copernicus Climate Change Service information (2026); Niño 3.4 and ONI (NOAA CPC); RMM index (Australian Bureau of Meteorology); ESA CCI/C3S SST (Copernicus Marine Service); NCEP GODAS (NOAA CPC).

Licence of the files in this folder: CC BY 4.0.
