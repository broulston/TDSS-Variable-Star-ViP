# TDSS Variable Star ViP

[![GitHub release](https://img.shields.io/github/release/broulston/TDSS-Variable-Star-ViP.svg)](https://github.com/broulston/TDSS-Variable-Star-ViP/releases/latest)
[![GitHub commits](https://img.shields.io/github/commits-since/broulston/TDSS-Variable-Star-ViP/v1.0.0.svg)](https://github.com/broulston/TDSS-Variable-Star-ViP/commits/master)
[![GitHub issues](https://img.shields.io/github/issues/broulston/TDSS-Variable-Star-ViP.svg)](https://github.com/broulston/TDSS-Variable-Star-ViP/issues)
[![license](https://img.shields.io/github/license/broulston/TDSS-Variable-Star-ViP.svg)](https://github.com/broulston/TDSS-Variable-Star-ViP/blob/master/license.txt)
[![Python Supported](https://img.shields.io/badge/Python%20Supported-3-brightgreen.svg)](conda)
[![Maintenance](https://img.shields.io/maintenance/yes/2022.svg)]()
[![Powered by Astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)

`TDSS ViP` creates one (1) plot for each of the variable stars in the TDSS program sample. These plots are designed to help summarize as much information about each star as possible, and to aid in classification. Each plot currently contains: 

![Vi_plot](./001.31303+01.92685_Vi.png?raw=true)

1. CRTS & ZTF light curves (in raw form if `logProb_Per > -5.0`, and in folded phase if `logProb_Per <= -5.0`)
2. Some general population plot based on if the star is periodic or not:
    * Plot from Palaversa et al. (2013) if periodic. This is `log10(P/days)` on the x-axis and `log10(Amp/mag)` on the y-axis with light curve skewness as the colorbar.
    * Generic _heat map_ of `a95` vs log10($\chi^2$) if star is non-periodic.
3. Color-Magnitude diagram based on SDSS colors and Gaia DR2 distances. This plot also uses an upperLim on the distance by assuming a space velocity of 600 km s$^{-1}$ and using GaiaDR2 proper motions to find the distance. This gives us a lowerLim on M$_{i}$, which is shown above the CMD and as a red arrow in the CMD plot.
4. SDSS spectrum with properties printed (this includes information from `PyHammer`)

These plots are created using the `ViP` code written by Ben Roulston (BU/SAO). In addition to these plots, the Vi code also creates a `.csv` file that contains information about the light curves for each star.

In order to run this Vi program, the following items are needed and **MUST** be downloaded/created/known before any figures can be made.

1. CSS or ZTFg or ZTFr light curves for every star.
2. File connecting the `CSS_ID` to the `RA` and `DEC` for each star. (A simple `.csv` that is just `[ra, dec, css_id]` works fine and is what is used here.
3. SDSS spectrum for every star.
    * If the star has been in a public data release, then the SDSS `.fits` format is fine (i.e. `spec-MJD-PLATE-FIBERID.fits`)
    * If the star has **NOT** been in a public data release (i.e. is proprietary) then you need to download the `spPlate-PLATE-MJD.fits` for each. Then pull out the spectra and save them in ASCII format.
4. Know where all data is stored and make sure the directories in the code are up to date (this is **VERY** important).
5. Property table which links **ALL** spALL data with Gaia DR2 data and CSS/ZTF data.
6. Installation of the `PyHammer` program for spectral classification. This should be run on all our objects and the output stored in the `sup_data` folder.

Once all of this is ready, the `Vi` program can be run. 

`Vi` also makes use of [PyHammer](https://github.com/BU-hammerTeam/PyHammer), a spectral classification tool developed by the @BU-hammerTeam at Boston University.
