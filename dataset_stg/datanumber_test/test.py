# %%
import os
import sys
sys.path.append((os.path.relpath(r'C:\Users\b4-12\Git\NCHC_ResIPy\resipy\src'))) # add here the relative path of the API folder
from resipy import Project

# %%
testdir = r'C:\Users\b4-12\Git\NCHC_ResIPy\datanumber_test'
k = Project(typ='R2') # create new Project object and use default working directory
k.createSurvey(os.path.join(testdir, 'test_5625.ohm'), ftype='BERT')