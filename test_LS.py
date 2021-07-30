import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.stats import f

from astropy.io import fits
from astropy.timeseries import LombScargle

from ResearchTools import LCtools

from astropy.table import Table
import astropy.units as u

import warnings
import tqdm
from subprocess import *
import os


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def freq2per(frequency, period_unit=u.d):
    return (frequency**-1).to(period_unit)


def per2freq(period, frequency_unit=u.microHertz):
    return (period**-1).to(frequency_unit)

all_LCs = np.where(TDSSprop.data['CSSLC'] & TDSSprop.data['ZTF_g_GroupID'] & TDSSprop.data['ZTF_r_GroupID'])[0]

checkalias = False
log10FAP = -10.0

prop_id = 12 # all_LCs[0]
ROW = TDSSprop.data[prop_id]

object_ra = ROW['ra']
object_dec = ROW['dec']
ra_string = '{:0>9.5f}'.format(object_ra)
dec_string = '{:0=+9.5f}'.format(object_dec)

is_CSS = ROW['CSSLC']
is_ZTF_g = np.isfinite(ROW['ZTF_g_GroupID'])
is_ZTF_r = np.isfinite(ROW['ZTF_r_GroupID'])

if is_CSS:
    lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
    CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
    CSS_lc_data.sort('mjd')
if is_ZTF_g:
    ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['GroupID'] == ROW['ZTF_g_GroupID'])]['mjd', 'mag', 'magerr']
    ZTF_g_lc_data.sort('mjd')
if is_ZTF_r:
    ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['GroupID'] == ROW['ZTF_r_GroupID'])]['mjd', 'mag', 'magerr']
    ZTF_r_lc_data.sort('mjd')


flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data, fltRange=5.0)

goodQualIndex = np.where(flc_data['QualFlag'] == True)[0]
lc_mjd = flc_data['mjd'][goodQualIndex]
lc_mag = flc_data['mag'][goodQualIndex]
lc_err = flc_data['magerr'][goodQualIndex]

t_days = lc_mjd * u.day
y_mags = lc_mag * u.mag
dy_mags = lc_err * u.mag

maxP = 1000.0 * u.d
minP = 0.01 * u.d

maximum_frequency = (minP)**-1
minimum_frequency = (maxP)**-1

freq_grid = np.linspace(0, maximum_frequency.value, num=250001)[1:] / u.d

ls_window = LombScargle(t_days, np.ones(y_mags.size), fit_mean=False, center_data=False)
window_power = ls_window.power(frequency=freq_grid)
window_power[~np.isfinite(window_power)] = 0
window_FAP_power_peak = np.nanstd(window_power).value * 4

# title = 'RA: {!s} DEC: {!s} Window Function Power Spectrum'.format(ra_string, dec_string)
# LCtools.plot_powerspec(freq_grid, window_power, FAP_power_peak=window_FAP_power_peak, title=title)

ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True)
power = ls.power(frequency=freq_grid)
logFAP_limit = -10
FAP_power_peak = ls.false_alarm_level(10**logFAP_limit)
df = (1 * u.d)**-1

# title = 'RA: {!s} DEC: {!s} Power Spectrum'.format(ra_string, dec_string)
# LCtools.plot_powerspec(freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

# where_period = np.where(window_power > window_FAP_power_peak)[0]
# grouped_where_period = group_consecutives(where_period)
# for ii in grouped_where_period:
#     print(period[ii].mean(), period[ii].std())

# per2freq(1.12574581 * u.d)

# importlib.reload(LCtools)
#  prop_out_dir, CSS_LC_plot_dir, ZTF_LC_plot_dir, Vi_plots_di
# lc_dir = ZTF_LC_plot_dir + "r/"
P = freq2per(freq_grid[np.argmax(power)]).value  # 1.12574581
P *= 2
P = freq2per(freq_grid[np.argmax(power)] - df).value
# title = "RA: {!s} DEC: {!s} $|$ P = {!s}d $|$ $m$ = {!s}".format(ra_string, dec_string, np.round(test_P, 7), np.round(mean_mag, 2))
title = "RA: {!s} DEC: {!s}".format(ra_string, dec_string)
LCtools.plot_LC_analysis(flc_data, P, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
# plt.savefig(f"{lc_dir}/{ra_string}{dec_string}_full_LC_analysis.pdf", dpi=600)
plt.show()
plt.clf()
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
title = "RA: {!s} DEC: {!s}".format(ra_string, dec_string)
LCtools.plt_any_lc(flc_data, P, is_Periodic=True, ax=None, fig=fig, title=title, phasebin=True)
# plt.savefig(f"{lc_dir}/{ra_string}{dec_string}_full_LC_analysis.pdf", dpi=600)
plt.show()
plt.clf()
plt.close()

importlib.reload(LCtools)
P = freq2per(freq_grid[np.argmax(power)]).value  # 1.12574581
title = "RA: {!s} DEC: {!s}".format(ra_string, dec_string)
LCtools.plot_LC_analysis_ALLaliases(flc_data, P, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
plt.show()
plt.clf()
plt.close()


from sklearn.cluster import MeanShift, estimate_bandwidth
# #############################################################################
sorted_P = np.flip(freq2per(freq_grid[np.argsort(power)]))
X = sorted_P[:100].reshape(-1,1)
# Compute clustering with MeanShift
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique, labels_index, labels_inverse, labels_inverse = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

P1 = X.flatten()[np.where(labels == np.argsort(labels_index)[0])[0][0]].value
P2 = X.flatten()[np.where(labels == np.argsort(labels_index)[1])[0][0]].value
P3 = X.flatten()[np.where(labels == np.argsort(labels_index)[2])[0][0]].value

importlib.reload(LCtools)
title = "RA: {!s} DEC: {!s}".format(ra_string, dec_string)
LCtools.plot_LC_analysis_Pal2013(flc_data, P1, P2, P3, freq_grid, power, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, df=df, title=title)
# plt.savefig(f"{ZTF_LC_plot_dir}r/{ra_string}{dec_string}_full_LC_analysis.pdf", dpi=600)
plt.show()
plt.clf()
plt.close()

P = P1 * 2 * u.d
t0 = t_days[np.argmin(y_mags)]

model_ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True, nterms=6)
# logFAP_limit = -10
# FAP_power_peak = ls.false_alarm_level(10**logFAP_limit)
# df = (1 * u.d)**-1

phase_fit = np.linspace(0, 1, 1000)
t_fit = (phase_fit / P**-1) + t0
y_fit = model_ls.model(t=t_fit, frequency=P**-1)

resid = (((y_mags - model_ls.model(t=t_days, frequency=P**-1)) / dy_mags)**2).sum() * (1 / (t_days.size - 1))


phase = ((t_days - t0) * P**-1) % 1

# model_ls.model_parameters(frequency=P**-1, units=True)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(phase, y_mags.value, dy_mags.value, c='g', marker=".", ls='None', mew=0, capsize=0, elinewidth=1.5)
ax.errorbar(phase + 1, y_mags.value, dy_mags.value, c='g', marker=".", ls='None', mew=0, capsize=0, elinewidth=1.5)
ax.plot(phase_fit, y_fit, color='b')
ax.plot(phase_fit + 1, y_fit, color='b')
ax.invert_yaxis()
ax.set(xlabel='phase',
       ylabel='magnitude',
       title='phased data at frequency={0:.2f}'.format(P**-1))

plt.show()
plt.clf()
plt.close()


def createHarmonicFrac(Nmax=4):
    hold = np.arange(2.0, Nmax + 1.0)
    return np.sort(np.hstack(([1], hold, hold**-1)))


def test_alias(P0, harmonics, t_days, y_mags, dy_mags, nterms=6):
    P0 = P0 * u.d
    model_ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True, nterms=nterms)

    resids = np.zeros(harmonics.size) * np.nan
    for ii, harm in enumerate(harmonics):
        P_test = harm * P0
        resids[ii] = np.nansum((((y_mags - model_ls.model(t=t_days, frequency=P_test**-1)) / dy_mags)**2)) * (1 / (t_days.size - 1))

    best_harmonic = harmonics[np.nanargmin(resids)]
    P = P0 * best_harmonic
    return P, best_harmonic, resids


P0 = P
harmonics = createHarmonicFrac(Nmax=4)
Pbest, best_harmonic, resids = test_alias(P0, harmonics, t_days, y_mags, dy_mags, nterms=3)

plot_P = 2.0 * Pbest

t0 = t_days[np.argmin(y_mags)]
model_ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True, nterms=np.random.randint(low=1, high=7))

phase_fit = np.linspace(0, 1, 1000)
t_fit = (phase_fit / plot_P**-1) + t0
y_fit = model_ls.model(t=t_fit, frequency=plot_P**-1)

data = [t_days.value, y_mags.value, dy_mags.value]
AFD_data = LCtools.AFD(data, plot_P.value, alpha=0.99, Nmax=6, checkalias=False)

phase = ((t_days - t0) * plot_P**-1) % 1

fig, ax = plt.subplots(figsize=(12, 2))
ax.errorbar(phase, y_mags.value, dy_mags.value, c='k', marker=".", ls='None', mew=0, capsize=0, elinewidth=1.5)
ax.errorbar(phase + 1, y_mags.value, dy_mags.value, c='k', marker=".", ls='None', mew=0, capsize=0, elinewidth=1.5)
ax.plot(phase_fit, y_fit, color='r')
ax.plot(phase_fit + 1, y_fit, color='r')
ax.invert_yaxis()
ax.set(xlabel='phase',
       ylabel='magnitude',
       title='phased data at f={!s} $|$ P={!s}'.format(np.round((plot_P**-1).to(u.microHertz), 6), np.round(plot_P, 6)))

plt.tight_layout()
plt.show()
plt.clf()
plt.close()