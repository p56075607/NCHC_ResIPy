# read tr.csv and interpolate the missing values
# output: tr_intp.csv
import pandas as pd
import numpy as np

def interpolate_missing_values(input_file, output_file):
    # Read the input CSV file : x,y,z
    # x,y,z
    # 0,0,186
    # 10,0,185
    # 60,0,184
    # 80,0,183
    # 100,0,177.4
    # 110,0,177.237
    # ...

    df = pd.read_csv(input_file)
    # Interpolate the missing values
    x = np.linspace(0,1000,101)
    y = np.zeros(101)
    z = np.interp(x, df['x'], df['z'])
    df_intp = pd.DataFrame({'x':x, 'y':y, 'z':z})
    
    # Save the interpolated data to a new CSV file
    df_intp.to_csv(output_file, index=False)
   

# Example usage
interpolate_missing_values(r'STG_test\tr.csv', r'STG_test\tr_intp.csv')