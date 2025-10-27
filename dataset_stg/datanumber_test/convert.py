# %%
import sys
sys.path.append(r'C:\Users\b4-12\Git\masterdeg_programs\pyGIMLi\ToolBox')
from convertURF import convertURF
from os.path import join
from os import listdir

urf_path = r'C:\Users\b4-12\Git\NCHC_ResIPy\datanumber_test'
urffiles = [_ for _ in listdir(urf_path) if _.endswith('.urf')]

for i,urf_file_name in enumerate(urffiles):
    ohm_file_name = convertURF(join(urf_path,urf_file_name),has_trn = False)