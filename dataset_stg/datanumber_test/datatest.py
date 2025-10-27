# %%
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"

import numpy as np

# Create a list of data
data_num = [180000, 90000, 45000, 22500, 11250, 5625]
memory_require = [10.2289,5.076, 2.612, 1.271, 0.638,0.319] # Gb
NCHC_processing_time = [4884, 4077,974,773,124,107] # s
Local_processing_time = [6661, 3209,838,713,114,96] # s

fig, ax1 = plt.subplots(figsize=(6,3))
ax1.plot(data_num, NCHC_processing_time, 'ro-',label='NCHC')
ax1.plot(data_num, Local_processing_time, 'go--',label='Local')
ax1.set_ylabel('Processing time (s)')
ax1.tick_params('y')
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()  # Adjust layout to make room for the rotated date labels
# ax2 = ax1.twinx()

# ax2.plot(data_num, memory_require, 'o-',color='gray',alpha=0.5)


# ax2.set_ylabel('Memory requirement (Gb)', color='gray')
# ax2.tick_params('y', colors='gray')
ax1.set_xticks(data_num)
ax1.grid(linestyle='--',linewidth=0.5)
ax1.set_xlabel('Data number')
ax1.legend(loc='upper left')
plt.title('Processing time and memory requirement')

fig.savefig(r'Z:\Downloads\datanumber_test\datanumber_test.png',dpi=300,bbox_inches='tight')