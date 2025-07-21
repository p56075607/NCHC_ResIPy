import pandas as pd
import numpy as np

def customStingParser(fname):
    """
    自定義的 .stg 文件解析器，修復了原始 stingParser 的 IndexError 問題
    """
    df_raw = pd.read_csv(fname, skipinitialspace=True, skiprows=3, header=None)
    elec_x =  np.concatenate(df_raw.iloc[:,[9,12,15,18]].values)
    elec_y =  np.concatenate(df_raw.iloc[:,[10,13,16,19]].values)
    elec_z =  np.concatenate(df_raw.iloc[:,[11,14,17,20]].values)
    elec_raw = np.unique(np.column_stack((elec_x,elec_y,elec_z)), axis=0)
    
    #detect 2D or 3D
    survey_type = '2D' if len(np.unique(elec_raw[:,1])) == 1 else '3D'
    
    if survey_type == '2D':
        elec = elec_raw[elec_raw[:,0].argsort(kind='mergesort')]# final electrode array
        a_f = [0]*len(df_raw)
        b_f = [0]*len(df_raw)
        n_f = [0]*len(df_raw)
        m_f = [0]*len(df_raw)
        
        for i in range(len(df_raw)):
            a_f[i] = list(elec[:,0]).index(df_raw.iloc[i,[9]].values)+1
            b_f[i] = list(elec[:,0]).index(df_raw.iloc[i,[12]].values)+1
            m_f[i] = list(elec[:,0]).index(df_raw.iloc[i,[15]].values)+1
            n_f[i] = list(elec[:,0]).index(df_raw.iloc[i,[18]].values)+1
    
    else: # 3D case
        elecdf = pd.DataFrame(elec_raw[elec_raw[:,1].argsort(kind='mergesort')]).rename(columns={0:'x',1:'y',2:'z'})
        # organize 3D electrodes
        elecdf_groups = elecdf.groupby('y', sort=False, as_index=False)
        elecdf_lines = [elecdf_groups.get_group(x).copy() for x in elecdf_groups.groups]
        
        ######### NOT SURE ABOUT BELOW - are electrodes laid out like a snake? ##########
        m = 0
        while 2*m + 1 <= len(elecdf_lines) - 1: # electrodes are laid out like a snake - although not sure if this is correct
            i = 2*m + 1 # index of odd lines
            elecdf_lines[i]['x'] = elecdf_lines[i]['x'].values[::-1]
            m += 1
        ######### NOT SURE ABOUT ABOVE #########
        
        elec = np.concatenate(elecdf_lines) # final electrode array
        
        lines = np.unique(elecdf.y) # basically saying what is the y val of each line    
        
        # for final array
        a_f = []
        b_f = []
        m_f = []
        n_f = []
        
        # positions of ABMN
        array_A = df_raw.iloc[:,9:12].rename(columns={9:'x',10:'y',11:'z'})
        array_B = df_raw.iloc[:,12:15].rename(columns={12:'x',13:'y',14:'z'})
        array_M = df_raw.iloc[:,15:18].rename(columns={15:'x',16:'y',17:'z'})
        array_N = df_raw.iloc[:,18:21].rename(columns={18:'x',19:'y',20:'z'})
        
        # building A locs/labels
        array_A_groups = array_A.groupby('y', sort=False, as_index=False)
        array_A_lines = [array_A_groups.get_group(x) for x in array_A_groups.groups]
        # which lines
        for line in array_A_lines:
            line_num = np.where(lines == line['y'].iloc[0])[0][0]
            a = [0]*len(line)
            for i in range(len(line)):
                a[i] = elecdf_lines[line_num]['x'][elecdf_lines[line_num]['x'] == line['x'].iloc[i]].index[0] + 1
            a_f.extend(a)
        
        # building B locs/labels
        array_B_groups = array_B.groupby('y', sort=False, as_index=False)
        array_B_lines = [array_B_groups.get_group(x) for x in array_B_groups.groups]
        # which lines
        for line in array_B_lines:
            line_num = np.where(lines == line['y'].iloc[0])[0][0]
            b = [0]*len(line)
            for i in range(len(line)):
                b[i] = elecdf_lines[line_num]['x'][elecdf_lines[line_num]['x'] == line['x'].iloc[i]].index[0] + 1
            b_f.extend(b)
        
        # building M locs/labels
        array_M_groups = array_M.groupby('y', sort=False, as_index=False)
        array_M_lines = [array_M_groups.get_group(x) for x in array_M_groups.groups]
        # which lines
        for line in array_M_lines:
            line_num = np.where(lines == line['y'].iloc[0])[0][0]
            m = [0]*len(line)
            for i in range(len(line)):
                m[i] = elecdf_lines[line_num]['x'][elecdf_lines[line_num]['x'] == line['x'].iloc[i]].index[0] + 1
            m_f.extend(m)
        
        # building N locs/labels
        array_N_groups = array_N.groupby('y', sort=False, as_index=False)
        array_N_lines = [array_N_groups.get_group(x) for x in array_N_groups.groups]
        # which lines
        for line in array_N_lines:
            line_num = np.where(lines == line['y'].iloc[0])[0][0]
            n = [0]*len(line)
            for i in range(len(line)):
                n[i] = elecdf_lines[line_num]['x'][elecdf_lines[line_num]['x'] == line['x'].iloc[i]].index[0] + 1
            n_f.extend(n)
    
    #build df
    df = pd.DataFrame()
    df['a'] = np.array(a_f)
    df['b'] = np.array(b_f)
    df['n'] = np.array(n_f)
    df['m'] = np.array(m_f)
    df['resist']=df_raw.iloc[:,4]

    #detecting IP - 修復了這裡的 IndexError 問題
    # 檢查是否有足夠的列來包含 IP 資料
    if df_raw.shape[1] > 21 and 'IP' in str(df_raw.iloc[0,21]): 
        # convert into chargeability via equation 1 in Mwakanyamale et al (2012)
        Vs_index = [24 + i for i in range(6)]
        #lookup table for time constants and corresponding time slot lengths 
        timeslot = {
            0:100,
            1:130, 
            2:260,
            4:540,
            8:1040, 
            }
        mrad = [0.0]*len(df_raw)
        for i in range(len(df_raw)):
            VsVp = np.asarray(df_raw.iloc[i,Vs_index], dtype=float) #secondary/primary voltage 
            tr = df_raw.iloc[i,4] # transfer resistance 
            Ia = df_raw.iloc[i,6]*1e-3 # current (amps)
            Vp = tr * Ia # potential voltage under current (in volts)
            Vs = Vp*VsVp*1e3 #secondary voltage in milli volts
            tt = int(df_raw.iloc[i,23]*1e-3) # total time measuring (seconds)
            if tt in timeslot.keys():
                dt = timeslot[tt]*1e-3
            else: 
                dt = 260e-3# change in time per slot 
            # integrate Vs to get chargeability 
            _T = np.abs(np.diff(Vs)) * dt * 0.5 # area of triangles under the curve 
            _C = np.abs(Vs[1:])*dt # area of colums
            Area = np.sum(_T*_C)
            Ma = ((1/(6*dt))*(Area/Vp))
            mrad[i] = Ma*-1 

        df['ip'] = mrad  
    else:
        df['ip'] = [0]*len(df_raw)
    
    #for pole-pole and pole-dipole arrays
    elec[elec > 9999] = 999999
    elec[elec < -9999] = -999999
    df = df.query('a!=b & b!=m & m!=n & a!=m & a!=n & b!=n').reset_index().drop(columns='index') # removing data where ABMN overlap
    
    return elec, df 