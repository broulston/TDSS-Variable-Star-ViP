def test_freqGrid_auto(lc_data, minP=0.1, maxP=100.0):
        goodQualIndex = np.where(lc_data['QualFlag'] == True)[0]
        lc_mjd = lc_data['mjd'][goodQualIndex]
        lc_mag = lc_data['mag'][goodQualIndex]
        lc_err = lc_data['magerr'][goodQualIndex]

        t_days = lc_mjd * u.day
        y_mags = lc_mag * u.mag
        dy_mags = lc_err * u.mag

        # window_y = np.ones_like(t_days)
        # window_ls = LombScargle(t_days, window_y, fit_mean=False, center_data=False, nterms=1)
        # frequency, power = window_ls.autopower(minimum_frequency=0/u.d, maximum_frequency=maximum_frequency)
        # plt.plot(frequency, power, lw=0.5, c='k')

        # plt.hist(np.diff(t_days[np.argsort(t_days)]).value)

        # nterms = 1
        # maxP = 1000.0 * u.d
        # minP = 0.1 * u.d

        # maximum_frequency = (minP)**-1
        # # minimum_frequency = (maxP)**-1

        # frequency = np.linspace(0, maximum_frequency.value, num=100001)[1:] / u.d

        # ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True)
        # power = ls.power(frequency=frequency)
        # FAP = ls.false_alarm_probability(power)

        ls = LombScargle(t_days, y_mags, dy_mags)

        maximum_frequency = (minP * u.day)**-1
        minimum_frequency = (maxP * u.day)**-1

        frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
        return frequency, power
        # FAP = ls.false_alarm_probability(power)


def test_freqGrid_set(lc_data, minP=0.1, maxP=100.0, Nf=250000):
        goodQualIndex = np.where(lc_data['QualFlag'] == True)[0]
        lc_mjd = lc_data['mjd'][goodQualIndex]
        lc_mag = lc_data['mag'][goodQualIndex]
        lc_err = lc_data['magerr'][goodQualIndex]

        t_days = lc_mjd * u.day
        y_mags = lc_mag * u.mag
        dy_mags = lc_err * u.mag

        # window_y = np.ones_like(t_days)
        # window_ls = LombScargle(t_days, window_y, fit_mean=False, center_data=False, nterms=1)
        # frequency, power = window_ls.autopower(minimum_frequency=0/u.d, maximum_frequency=maximum_frequency)
        # plt.plot(frequency, power, lw=0.5, c='k')

        # plt.hist(np.diff(t_days[np.argsort(t_days)]).value)

        # nterms = 1
        # maxP = 1000.0 * u.d
        # minP = 0.1 * u.d

        maximum_frequency = (minP * u.d)**-1
        # # minimum_frequency = (maxP)**-1

        frequency = np.linspace(0, maximum_frequency.value, num=Nf+1)[1:] / u.d

        ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True)
        power = ls.power(frequency=frequency)
        # FAP = ls.false_alarm_probability(power)

        # ls = LombScargle(t_days, y_mags, dy_mags)

        # maximum_frequency = (minP * u.day)**-1
        # minimum_frequency = (maxP * u.day)**-1

        # frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
        return frequency, power


CSS_freq_data = np.zeros((len(TDSSprop.data), 5)) * np.nan
ZTFg_freq_data = np.zeros((len(TDSSprop.data), 5)) * np.nan
ZTFr_freq_data = np.zeros((len(TDSSprop.data), 5)) * np.nan

for prop_id, ROW in enumerate(tqdm.tqdm(TDSSprop.data)):
    if prop_id < prop_id_last:
        properties[prop_id]['ViCompleted'] = 1
        continue
    if not ROW['CSS_or_ZTF']:
        properties[prop_id]['ViCompleted'] = 1
        continue

    any_good_LC = (ROW['CSS_Nepochs'] >= Nepochs_required) | (ROW['ZTF_g_Nepochs'] >= Nepochs_required) | (ROW['ZTF_r_Nepochs'] >= Nepochs_required)
    if not any_good_LC:
        continue
    #

    object_ra = ROW['ra']
    object_dec = ROW['dec']
    ra_string = '{:0>9.5f}'.format(object_ra)
    dec_string = '{:0=+9.5f}'.format(object_dec)

    is_CSS = ROW['CSSLC']
    is_ZTF_g = np.isfinite(ROW['ZTF_g_GroupID'])
    is_ZTF_r = np.isfinite(ROW['ZTF_r_GroupID'])

    ra_string = '{:0>9.5f}'.format(ROW['ra'])
    dec_string = '{:0=+9.5f}'.format(ROW['dec'])

    if is_CSS:
        try:
            lc_file = CSS_LC_dir+str(ROW['CSSID'])+'.dat'
            CSS_lc_data = Table.read(lc_file, format='ascii', names=['mjd', 'mag', 'magerr'])
            if len(CSS_lc_data)>=Nepochs_required:
                popt = np.array([-2.61242938e+01,  1.93636204e+00,  4.45971381e-01, -6.49419310e-02, 2.99231126e-03,  2.40758201e-01, -2.40805035e-01])
                magerr_resid_mean = 0.008825118765717422
                shift_const = 1.5 * magerr_resid_mean
                pred_magerr = low_order_poly(CSS_lc_data['mag'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
                bad_err_index = np.where(CSS_lc_data['magerr'] - (pred_magerr - shift_const) < 0.0)[0]
                CSS_lc_data['magerr'][bad_err_index] = pred_magerr[bad_err_index]

                CSS_flc_data, LC_stat_properties = LCtools.process_LC(CSS_lc_data.copy(), fltRange=5.0)
                CSS_frequency = test_freqGrid_auto(CSS_flc_data, minP=minP, maxP=maxP)
                CSS_freq_data[prop_id, 0] = CSS_frequency.size
                CSS_freq_data[prop_id, 1] = CSS_frequency.min().value
                CSS_freq_data[prop_id, 2] = CSS_frequency.max().value
                CSS_freq_data[prop_id, 3] = CSS_frequency.diff().mean().value
                CSS_freq_data[prop_id, 4] = CSS_frequency.diff().std().value
        except:
            pass
    if is_ZTF_g:
        ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs['GroupID'] == ROW['ZTF_g_GroupID'])]['mjd', 'mag', 'magerr']
        if len(ZTF_g_lc_data)>=Nepochs_required:
            ZTF_gflc_data, LC_stat_properties = LCtools.process_LC(ZTF_g_lc_data.copy(), fltRange=5.0)
            ZTFg_frequency = test_freqGrid_auto(ZTF_gflc_data, minP=minP, maxP=maxP)
            ZTFg_freq_data[prop_id, 0] = ZTFg_frequency.size
            ZTFg_freq_data[prop_id, 1] = ZTFg_frequency.min().value
            ZTFg_freq_data[prop_id, 2] = ZTFg_frequency.max().value
            ZTFg_freq_data[prop_id, 3] = ZTFg_frequency.diff().mean().value
            ZTFg_freq_data[prop_id, 4] = ZTFg_frequency.diff().std().value
    if is_ZTF_r:
        ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs['GroupID'] == ROW['ZTF_r_GroupID'])]['mjd', 'mag', 'magerr']
        if len(ZTF_r_lc_data)>=Nepochs_required:
            ZTF_rflc_data, LC_stat_properties = LCtools.process_LC(ZTF_r_lc_data.copy(), fltRange=5.0)
            ZTFr_frequency, ZTFr_power = test_freqGrid_auto(ZTF_rflc_data, minP=minP, maxP=maxP)
            ZTFr_freq_data[prop_id, 0] = ZTFr_frequency.size
            ZTFr_freq_data[prop_id, 1] = ZTFr_frequency.min().value
            ZTFr_freq_data[prop_id, 2] = ZTFr_frequency.max().value
            ZTFr_freq_data[prop_id, 3] = ZTFr_frequency.diff().mean().value
            ZTFr_freq_data[prop_id, 4] = ZTFr_frequency.diff().std().value



%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_auto(ZTF_rflc_data)
%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_set(ZTF_rflc_data, Nf=100000)
%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_set(ZTF_rflc_data, Nf=200000)

%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_auto(CSS_flc_data)
%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_set(CSS_flc_data, Nf=100000)
%timeit ZTFr_frequency, ZTFr_power = test_freqGrid_set(CSS_flc_data, Nf=200000)


