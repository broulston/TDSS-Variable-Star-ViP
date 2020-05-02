import numpy as np
from astropy.table import Table, hstack
from astropy.io import fits
import numpy.core.defchararray as np_f

results = Table.read("pyhammer_result_0309_vs_0408.fits")

PyHammerSpecType_1 = np_f.strip(results['PyHammerSpecType_1'].data).astype(str)
PyHammerSpecType_2 = np_f.strip(results['PyHammerSpecType_2'].data).astype(str)

results.replace_column('PyHammerSpecType_1', PyHammerSpecType_1)
results.replace_column('PyHammerSpecType_2', PyHammerSpecType_2)

where_changed = PyHammerSpecType_1!=PyHammerSpecType_2

changed_table = results[['PyHammerSpecType_1', 'PyHammerSpecType_2']][where_changed]

changed_table = results[['ra_2', 'dec_2', 'plate_2', 'mjd_2', 'fiber_2', 'PyHammerSpecType_1', 'PyHammerSpecType_2', 'PyHammerRV_2']][where_changed]
names = ('ra_2', 'dec_2', 'plate_2', 'mjd_2', 'fiber_2')
new_names = ('ra', 'dec', 'plate', 'mjd', 'fiber')
changed_table.rename_columns(names, new_names)

type1_binary = [i for i,item in enumerate(changed_table['PyHammerSpecType_1'].data) if "+" in item]
type2_binary = [i for i,item in enumerate(changed_table['PyHammerSpecType_2'].data) if "+" in item]