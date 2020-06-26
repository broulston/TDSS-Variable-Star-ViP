import numpy as np
from astropy.table import Table, hstack
from astropy.io import fits
import numpy.core.defchararray as np_f
from tqdm import tqdm
import VarStar_Vi_plot_functions as vi

varstar_data = fits.open('/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/PROGRAM_SAMPLE/2020-06-24/FINAL_FILES/TDSS_SES+PREV_DR16DR12griLT20_GAIADR2_Drake2014PerVar_CSSID.fits')

all_ra = varstar_data[1].data['ra']
all_dec = varstar_data[1].data['dec']
all_plates = varstar_data[1].data['plate']
all_mjds = varstar_data[1].data['mjd']
all_fiberids = varstar_data[1].data['fiber']

pyhammer_result_filename = "PyHammerResults_VarStar_06-24-2020"

pyhammer_file = np.genfromtxt("sup_data/"+pyhammer_result_filename+".csv", dtype="U", comments="#", delimiter=",")
pyhammer_plate_mjd_fiberid = np_f.split(np_f.replace(np_f.replace(np_f.replace(pyhammer_file[:,0], "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/spec-", ""), "-", " "), ".fits", ""))

pyhammer_plate_mjd_fiberid_array = np.empty((pyhammer_plate_mjd_fiberid.size, 3), dtype=int)
for ii in tqdm(range(pyhammer_plate_mjd_fiberid_array.shape[0])):
    pyhammer_plate_mjd_fiberid_array[ii, :] = pyhammer_plate_mjd_fiberid[ii]

PyHammerSpecType = pyhammer_file[:,3]
PyHammerRV = pyhammer_file[:,2].astype(np.float64)

specInd = Table.read('sup_data/spectralIndices_VarStar_06-24-2020.csv', format='csv')
path_to_be_remvoed = "/Users/benjaminroulston/Dropbox/Research/TDSS/Variable_Stars/HARD_COPY_ORGINAL_DATA/SDSS_spec/02-26-2020/SDSSspec/"
specInd_pyhammer_plate_mjd_fiberid = np_f.split(
                                        np_f.replace(
                                                     np_f.replace(
                                                                  np_f.replace(
                                                                               np_f.replace(
                                                                                            specInd['#Filename'].data,
                                                                                            path_to_be_remvoed, ""),
                                                                               "spec-", ""),
                                                                  "-", " "),
                                                     ".fits", "")
                                        )
specInd_pyhammer_plate_mjd_fiberid_array = np.empty((specInd_pyhammer_plate_mjd_fiberid.size, 3), dtype=int)
for ii in tqdm(range(specInd_pyhammer_plate_mjd_fiberid_array.shape[0])):
    specInd_pyhammer_plate_mjd_fiberid_array[ii, :] = specInd_pyhammer_plate_mjd_fiberid[ii]

match_pyhammer_ra = []
match_pyhammer_dec = []

match_pyhammer_plates = []
match_pyhammer_mjds = []
match_pyhammer_fiberids = []

match_PyHammerSpecType = []
match_PyHammerRV = []

total_match_index = []
specInd_match_index = []
for ii in tqdm(range(all_ra.shape[0])):
    try:
      match_index = np.where((pyhammer_plate_mjd_fiberid_array[:, 0] == all_plates[ii])
                             & (pyhammer_plate_mjd_fiberid_array[:, 1] == all_mjds[ii])
                             & (pyhammer_plate_mjd_fiberid_array[:, 2] == all_fiberids[ii])
                             )[0][0]
      total_match_index.append(match_index)
      match_pyhammer_ra.append(all_ra[ii])
      match_pyhammer_dec.append(all_dec[ii])

      match_pyhammer_plates.append(all_plates[ii])
      match_pyhammer_mjds.append(all_mjds[ii])
      match_pyhammer_fiberids.append(all_fiberids[ii])

      match_PyHammerSpecType.append(PyHammerSpecType[match_index])
      match_PyHammerRV.append(PyHammerRV[match_index])

      specInd_match_index.append(np.where((specInd_pyhammer_plate_mjd_fiberid_array[:, 0] == all_plates[ii])
                                     & (specInd_pyhammer_plate_mjd_fiberid_array[:, 1] == all_mjds[ii])
                                     & (specInd_pyhammer_plate_mjd_fiberid_array[:, 2] == all_fiberids[ii])
                                     )[0][0])
    except:
      print(f"Bad match: spec={all_plates[ii]}-{all_mjds[ii]}-{all_fiberids[ii]}.fits")


match_pyhammer_ra = np.array(match_pyhammer_ra)
match_pyhammer_dec = np.array(match_pyhammer_dec)
match_pyhammer_plates = np.array(match_pyhammer_plates).astype(int)
match_pyhammer_mjds = np.array(match_pyhammer_mjds).astype(int)
match_pyhammer_fiberids = np.array(match_pyhammer_fiberids).astype(int)

match_PyHammerSpecType = np.array(match_PyHammerSpecType)
match_PyHammerRV = np.array(match_PyHammerRV)

total_match_index = np.array(total_match_index)
specInd_match_index = np.array(specInd_match_index)

pyhammer_table = Table()

pyhammer_table.add_column(match_pyhammer_ra, name='ra')
pyhammer_table.add_column(match_pyhammer_dec, name='dec')

pyhammer_table.add_column(match_pyhammer_plates, name='plate')
pyhammer_table.add_column(match_pyhammer_mjds, name='mjd')
pyhammer_table.add_column(match_pyhammer_fiberids, name='fiber')

pyhammer_table.add_column(match_PyHammerSpecType, name='PyHammerSpecType')
pyhammer_table.add_column(match_PyHammerRV, name='PyHammerRV')

pyhammer_table = hstack([pyhammer_table, specInd[specInd.columns.keys()[1:-1]][specInd_match_index]])

pyhammer_table.write("sup_data/"+pyhammer_result_filename+".fits", format='fits', overwrite=True)




 # u, indices, indices_inverse, indices_counts = np.unique(total_match_index, return_index=True, return_inverse=True, return_counts=True)

# pyhammer_ra = []
# pyhammer_dec = []

# pyhammer_plates = []
# pyhammer_mjds = []
# pyhammer_fiberids = []



# for ii in range(pyhammer_plate_mjd_fiberid.shape[0]):
#     pyhammer_plates.append(pyhammer_plate_mjd_fiberid[ii][0])
#     pyhammer_mjds.append(pyhammer_plate_mjd_fiberid[ii][1])
#     pyhammer_fiberids.append(pyhammer_plate_mjd_fiberid[ii][2])

#     match_index = np.where((all_plates     == int(pyhammer_plate_mjd_fiberid[ii][0]))
#                            & (all_mjds     == int(pyhammer_plate_mjd_fiberid[ii][1]))
#                            & (all_fiberids == int(pyhammer_plate_mjd_fiberid[ii][2]))
#                            )[0][0]
#     pyhammer_ra.append(all_ra[match_index])
#     pyhammer_dec.append(all_dec[match_index])


# pyhammer_ra = np.array(pyhammer_ra)
# pyhammer_dec = np.array(pyhammer_dec)

# pyhammer_plates = np.array(pyhammer_plates).astype(int)
# pyhammer_mjds = np.array(pyhammer_mjds).astype(int)
# pyhammer_fiberids = np.array(pyhammer_fiberids).astype(int)

# pyhammer_table = Table()

# pyhammer_table.add_column(pyhammer_ra, name='ra')
# pyhammer_table.add_column(pyhammer_dec, name='dec')

# pyhammer_table.add_column(pyhammer_plates, name='plate')
# pyhammer_table.add_column(pyhammer_mjds, name='mjd')
# pyhammer_table.add_column(pyhammer_fiberids, name='fiber')

# pyhammer_table.add_column(pyhammer_file[:,3], name='PyHammerSpecType')
# pyhammer_table.add_column(pyhammer_file[:,2].astype(np.float64), name='PyHammerRV')


# pyhammer_table.write("sup_data/"+pyhammer_result_filename+".fits", format='fits', overwrite=True)

# pyhammer_table = hstack([pyhammer_table, specInd[specInd.columns.keys()[1:-1]]])
#********************************************************************************
#********************************************************************************
#********************************************************************************





# pyhammer_ra = []
# pyhammer_dec = []

# pyhammer_plates = []
# pyhammer_mjds = []
# pyhammer_fiberids = []

# for ii in range(pyhammer_plate_mjd_fiberid.shape[0]):
#     try:
#         match_index = np.where((all_plates   == int(pyhammer_plate_mjd_fiberid[ii][0]))
#                              & (all_mjds     == int(pyhammer_plate_mjd_fiberid[ii][1]))
#                              & (all_fiberids == int(pyhammer_plate_mjd_fiberid[ii][2]))
#                              )[0][0]
#         pyhammer_ra.append(all_ra[match_index])
#         pyhammer_dec.append(all_dec[match_index])
#     except:
#         pyhammer_ra.append(np.nan)
#         pyhammer_dec.append(np.nan)

#     pyhammer_plates.append(pyhammer_plate_mjd_fiberid[ii][0])
#     pyhammer_mjds.append(pyhammer_plate_mjd_fiberid[ii][1])
#     pyhammer_fiberids.append(pyhammer_plate_mjd_fiberid[ii][2])

# pyhammer_ra = np.array(pyhammer_ra)
# pyhammer_dec = np.array(pyhammer_dec)

# pyhammer_plates = np.array(pyhammer_plates).astype(int)
# pyhammer_mjds = np.array(pyhammer_mjds).astype(int)
# pyhammer_fiberids = np.array(pyhammer_fiberids).astype(int)

# specInd = specInd[specInd.columns.keys()[1:-1]]

# specInd.add_column(pyhammer_ra, name='ra', index=0)
# specInd.add_column(pyhammer_dec, name='dec', index=1)

# specInd.add_column(pyhammer_plates, name='plate', index=2)
# specInd.add_column(pyhammer_mjds, name='mjd', index=3)
# specInd.add_column(pyhammer_fiberids, name='fiber', index=4)

# specInd.write("sup_data/"+pyhammer_result_filename+"_SpecIndices.fits", format='fits', overwrite=True)

