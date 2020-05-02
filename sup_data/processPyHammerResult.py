import numpy as np
from astropy.table import Table, hstack
from astropy.io import fits
import numpy.core.defchararray as np_f

import VarStar_Vi_plot_functions as vi

nbins=50
TDSSprop = vi.TDSSprop(nbins)

all_ra =TDSSprop.data['ra']
all_dec =TDSSprop.data['dec']
all_plates =TDSSprop.data['plate']
all_mjds = TDSSprop.data['mjd']
all_fiberids = TDSSprop.data['fiber']

pyhammer_result_filename = "PyHammerResults_VarStar_04-08-2020"

pyhammer_file = np.genfromtxt("sup_data/"+pyhammer_result_filename+".csv", dtype="U", comments="#", delimiter=",")
pyhammer_plate_mjd_fiberid = np_f.split(np_f.replace(np_f.replace(np_f.replace(pyhammer_file[:,0], "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/spec-", ""), "-", " "), ".fits", ""))

pyhammer_ra = []
pyhammer_dec = []

pyhammer_plates = []
pyhammer_mjds = []
pyhammer_fiberids = []

for ii in range(pyhammer_plate_mjd_fiberid.shape[0]):
    pyhammer_plates.append(pyhammer_plate_mjd_fiberid[ii][0])
    pyhammer_mjds.append(pyhammer_plate_mjd_fiberid[ii][1])
    pyhammer_fiberids.append(pyhammer_plate_mjd_fiberid[ii][2])

    match_index = np.where((all_plates     == int(pyhammer_plate_mjd_fiberid[ii][0]))
                           & (all_mjds     == int(pyhammer_plate_mjd_fiberid[ii][1]))
                           & (all_fiberids == int(pyhammer_plate_mjd_fiberid[ii][2]))
                           )[0][0]
    pyhammer_ra.append(all_ra[match_index])
    pyhammer_dec.append(all_dec[match_index])

pyhammer_ra = np.array(pyhammer_ra)
pyhammer_dec = np.array(pyhammer_dec)

pyhammer_plates = np.array(pyhammer_plates).astype(int)
pyhammer_mjds = np.array(pyhammer_mjds).astype(int)
pyhammer_fiberids = np.array(pyhammer_fiberids).astype(int)

pyhammer_table = Table()

pyhammer_table.add_column(pyhammer_ra, name='ra')
pyhammer_table.add_column(pyhammer_dec, name='dec')

pyhammer_table.add_column(pyhammer_plates, name='plate')
pyhammer_table.add_column(pyhammer_mjds, name='mjd')
pyhammer_table.add_column(pyhammer_fiberids, name='fiber')

pyhammer_table.add_column(pyhammer_file[:,3], name='PyHammerSpecType')
pyhammer_table.add_column(pyhammer_file[:,2].astype(np.float64), name='PyHammerRV')


pyhammer_table.write("sup_data/"+pyhammer_result_filename+".fits", format='fits', overwrite=True)

pyhammer_table = hstack([pyhammer_table, specInd[specInd.columns.keys()[1:-1]]])
#********************************************************************************
#********************************************************************************
#********************************************************************************

specInd = Table.read('sup_data/spectralIndices_VarStar_04-08-2020.csv', format='csv')
pyhammer_plate_mjd_fiberid = np_f.split(np_f.replace(np_f.replace(np_f.replace(specInd['#Filename'].data, "spec-", ""), "-", " "), ".fits", ""))

pyhammer_ra = []
pyhammer_dec = []

pyhammer_plates = []
pyhammer_mjds = []
pyhammer_fiberids = []

for ii in range(pyhammer_plate_mjd_fiberid.shape[0]):
    pyhammer_plates.append(pyhammer_plate_mjd_fiberid[ii][0])
    pyhammer_mjds.append(pyhammer_plate_mjd_fiberid[ii][1])
    pyhammer_fiberids.append(pyhammer_plate_mjd_fiberid[ii][2])

    match_index = np.where((all_plates     == int(pyhammer_plate_mjd_fiberid[ii][0]))
                           & (all_mjds     == int(pyhammer_plate_mjd_fiberid[ii][1]))
                           & (all_fiberids == int(pyhammer_plate_mjd_fiberid[ii][2]))
                           )[0][0]
    pyhammer_ra.append(all_ra[match_index])
    pyhammer_dec.append(all_dec[match_index])

pyhammer_ra = np.array(pyhammer_ra)
pyhammer_dec = np.array(pyhammer_dec)

pyhammer_plates = np.array(pyhammer_plates).astype(int)
pyhammer_mjds = np.array(pyhammer_mjds).astype(int)
pyhammer_fiberids = np.array(pyhammer_fiberids).astype(int)

specInd = specInd[specInd.columns.keys()[1:-1]]

specInd.add_column(pyhammer_ra, name='ra', index=0)
specInd.add_column(pyhammer_dec, name='dec', index=1)

specInd.add_column(pyhammer_plates, name='plate', index=2)
specInd.add_column(pyhammer_mjds, name='mjd', index=3)
specInd.add_column(pyhammer_fiberids, name='fiber', index=4)

specInd.write("sup_data/"+pyhammer_result_filename+"_SpecIndices.fits", format='fits', overwrite=True)

