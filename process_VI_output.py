import numpy as np
from astropy.table import Table
from astropy import units as u


prop = Table.read("TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_05-05-2020.fits")

prop_col_names_prefix = ['CSS_', 'ZTF_g_', 'ZTF_r_']
prop_col_names = ['lc_id', 'P', 'logProb', 'Amp', 'Mt', 'a95',
                  'lc_skew', 'Chi2', 'brtcutoff', 'brt10per',
                  'fnt10per', 'fntcutoff', 'errmn', 'ferrmn',
                  'ngood', 'nrejects', 'nabove', 'nbelow',
                  'Tspan100', 'Tspan95', 'isAlias', 'time_whittened',
                  'VarStat', 'Con', 'm', 'b_lin', 'chi2_lin',
                  'a', 'b_quad', 'c', 'chi2_quad']


for ii in prop_col_names_prefix:
    mask_rows = prop[f'{ii}ngood']==0.0
    for jj in prop_col_names:
        prop_col_name_full = ii + jj
        prop.mask[prop_col_name_full][mask_rows] = True

prop.write("TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_VI_ALLPROP_05-05-2020.fits", format="fits", overwrite=True)
