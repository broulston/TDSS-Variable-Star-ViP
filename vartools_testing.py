result = check_output(vt_dir+'vartools -i '+lc_file+' -LS 0.1 10.0 0.1 5 0', shell=True)
result = np.array(result.split()[1:]).astype(np.float64)
periods = result[::4]
logprob = result[1::4]
LSperiodgramVal = result[2::4]
LSpeakSNR = result[3::4]


t0 = flc_mjd[flc_mag.idxmin()]
for period in periods:
    phase = ((flc_mjd - t0)/period) % 1

    plt.scatter(phase, flc_mag)
    plt.show()
    plt.clf()
    plt.close()
    model = 0.232*np.sin((2*np.pi*flc_mjd)/period)+flc_mag.mean()
    print('ChiSq: ',(((flc_mag - model)/flc_err)**2).sum())

    plt.scatter(phase, flc_mag - model)
    plt.show()
    plt.clf()
    plt.close()


from astropy.time import Time
