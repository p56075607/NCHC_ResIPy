#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERT時序監測資料處理主程式
Main Program for Time-lapse ERT Data Processing

功能包含：
- 完整的ERT資料處理流程
- 自動化批次處理

作者: CHEN CHUN
日期: 2025
"""

import sys
import argparse
from pathlib import Path
import yaml

# 添加當前目錄到Python路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import warnings
warnings.filterwarnings('ignore')

from resipy import Project

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

class ERTTimeSeriesProcessor:
    """時序ERT資料處理器"""
    
    def __init__(self, config_file='config.yaml'):
        """
        初始化處理器
        
        參數:
            config_file: 配置檔案路徑
        """
        self.config = self._load_config(config_file)
        self._setup_logging()
        self._setup_directories()
        self.ERT = None
        self.survey_count = 0
        
    def _load_config(self, config_file):
        """載入配置檔案"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置檔案 {config_file} 不存在")
        except yaml.YAMLError as e:
            raise ValueError(f"配置檔案格式錯誤: {e}")
    
    def _setup_logging(self):
        """設置日誌"""
        log_level = getattr(logging, self.config['logging']['level'])
        
        # 設置日誌格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 設置主要記錄器
        self.logger = logging.getLogger('ERTProcessor')
        self.logger.setLevel(log_level)
        self.logger.addHandler(console_handler)
        
        # 檔案處理器
        if self.config['logging']['save_log']:
            log_file = self.config['logging']['log_file']
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_directories(self):
        """建立輸出目錄"""
        output_dir = Path(self.config['data']['output_dir'])

        if output_dir.exists() and self.config.get('output', {}).get('clear_output_before_run', True):
            import shutil
            self.logger.info(f"清空現有輸出目錄: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                self.logger.info("輸出目錄已清空")
            except Exception as e:
                self.logger.warning(f"清空輸出目錄時發生錯誤: {str(e)}")

        output_dir.mkdir(exist_ok=True)
        
        # 建立子目錄
        subdirs = ['pseudo_plots', 'mesh_plots', 'result_plots', 
                   'error_plots', 'convergence_plots', 'numerical_data']
        
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        self.logger.info(f"輸出目錄已建立: {output_dir}")

    @staticmethod
    def convertURF(data_read_path = 'L1_1_m.urf', has_trn = False, trn_path = 'TaTun_ERT1.trn'):
        """
        將 urf 檔案轉換為 ohm 檔案，並支援地形內插
        
        參數:
            data_read_path: urf 檔案路徑
            has_trn: 是否使用地形檔案
            trn_path: 地形檔案路徑
        
        返回:
            data_write_path: 輸出的 ohm 檔案路徑
        """
        Trn = []
        trn_x = []
        trn_elevation = []
        
        if has_trn:
            print('Using terrain file name: ', trn_path)
            with open(trn_path, 'r') as read_obj:
                for i, line in enumerate(read_obj):
                    if i > 2:  # 跳過前3行（標題和單位行）
                        line = line.replace("\n", "").strip()
                        # 跳過空行
                        if not line:
                            continue
                        
                        values = line.split(',')
                        # 確保有足夠的值且數值有效
                        if len(values) >= 2:
                            try:
                                x_val = float(values[0].strip())
                                elev_val = float(values[1].strip())
                                
                                Trn.append(values)
                                # 為了內插，分別保存 x 坐標和高程
                                trn_x.append(x_val)
                                trn_elevation.append(elev_val)
                            except ValueError as e:
                                print(f"警告：跳過無效的數據行 {i+1}: {line} - 錯誤: {e}")
                        else:
                            print(f"警告：跳過格式不正確的行 {i+1}: {line}")

        # 檢查是否成功讀取了地形數據
        if has_trn and not trn_x:
            print("警告：沒有從地形檔案中讀取到有效數據！將使用 urf 檔案中的原始 z 坐標。")
            has_trn = False
        elif has_trn:
            print(f"成功從地形檔案中讀取了 {len(trn_x)} 個有效地形點")

        Trn_array = np.array(Trn, dtype=float) if Trn else np.array([])

        def nonblank_lines(f):
            for l in f:
                line = l.rstrip()
                if line:
                    yield line

        data_write_path = data_read_path[:-4] + ".ohm"

        print('The original urf file name: ', data_read_path)
        print('The output ohm file name: ', data_write_path)

        string_to_search1 = ":Geometry"
        string_to_search2 = ":Measurements"

        electrode_position = []
        resistivity_measurement = []
        Line = []
        with open(data_read_path, 'r') as read_obj:
            nonblank_lines_obj = nonblank_lines(read_obj)
            enu_read_obj = enumerate(nonblank_lines_obj)
            for i, line in enu_read_obj:
                Line.append(line.split(','))
        
        # 提取電極位置數據，跳過註解行
        geometry_start = Line.index([string_to_search1]) + 1
        measurements_start = Line.index([string_to_search2])
        
        for line in Line[geometry_start:measurements_start]:
            # 跳過以分號開頭的註解行
            if line[0].strip().startswith(';'):
                continue
            electrode_position.append(line)
        
        # 提取測量數據，跳過註解行
        for line in Line[measurements_start + 1:]:
            # 跳過以分號開頭的註解行
            if line[0].strip().startswith(';'):
                continue
            resistivity_measurement.append(line)
        
        electrode_position_array = np.array(electrode_position, dtype=float)

        with open(data_write_path, 'w') as write_obj:
            str = '%d # Number of electrodes\n' % (len(electrode_position))
            write_obj.write(str)
            write_obj.write('# x z position for each electrode\n')
            
            for i in range(len(electrode_position)):
                x_pos = electrode_position_array[i, 1]  # 電極的 x 坐標
                z_pos = electrode_position_array[i, 3]  # 電極的原始 z 坐標
                
                if has_trn and len(trn_x) > 0:
                    # 使用 numpy 的 interp 函數進行線性內插
                    # 如果 x_pos 超出了 trn_x 的範圍，使用最近的值
                    if x_pos < min(trn_x):
                        elevation = trn_elevation[trn_x.index(min(trn_x))]
                    elif x_pos > max(trn_x):
                        elevation = trn_elevation[trn_x.index(max(trn_x))]
                    else:
                        elevation = np.interp(x_pos, trn_x, trn_elevation)
                    
                    # 將原始 z 坐標加上內插的地形高程
                    # 注意：如果 z 是負值（例如地下深度），保持為負值並加上高程
                    final_z = elevation + z_pos
                    
                    print(f"電極 {i+1}: x={x_pos}, 原始z={z_pos}, 內插高程={elevation}, 最終z={final_z}")
                    str = '%s   %s\n' % (x_pos, final_z)
                else:
                    str = '%s   %s\n' % (electrode_position[i][1], electrode_position[i][3])
                
                write_obj.write(str)

            str = '%d # Number of data\n' % (len(resistivity_measurement))
            write_obj.write(str)
            write_obj.write('# a b m n r i Uerr\n')
            for i in range(len(resistivity_measurement)):
                # URF格式：A, B, M, N, V/I(ohm) - 只有5個欄位
                # OHM格式：a b m n r i Uerr - 需要7個欄位
                
                # 檢查並處理零值
                resistance = float(resistivity_measurement[i][4].strip())
                if resistance == 0.0:
                    print('Find ZERO value!! \n Change to 0.00000000001 at ', i)
                    resistance = 0.00000000001
                
                # 計算電流 (假設為1.0安培，可以根據實際情況調整)
                current = 1.0
                
                # 計算誤差 (假設為3%，可以根據實際情況調整)
                error_percent = 3.0
                
                # 寫入OHM格式：a b m n r i Uerr
                str = '%s   %s   %s   %s   %s   %s   %s\n' % (
                    resistivity_measurement[i][0].strip(),  # a
                    resistivity_measurement[i][1].strip(),  # b  
                    resistivity_measurement[i][2].strip(),  # m
                    resistivity_measurement[i][3].strip(),  # n
                    resistance,                             # r (resistance)
                    current,                               # i (current)
                    error_percent                          # Uerr (error percentage)
                )
                write_obj.write(str)

        return data_write_path

    
    def load_data(self):
        """載入資料"""
        input_dir = Path(self.config['data']['input_dir'])
        
        if not input_dir.exists():
            raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
        
        # 檢查檔案格式
        files = list(input_dir.glob('*'))
        data_files = []
        
        for file in files:
            if file.suffix in self.config['data']['supported_formats']:
                data_files.append(file)
        
        if not data_files:
            raise ValueError(f"在 {input_dir} 中找不到支援的資料檔案")
        
        self.logger.info(f"找到 {len(data_files)} 個資料檔案")
        
        # 檢查檔案格式一致性
        file_formats = [f.suffix for f in data_files]
        
        if not all(fmt == file_formats[0] for fmt in file_formats):
            self.logger.warning("檔案格式不一致，將分別處理")
        
        # 處理不同格式的檔案
        processed_files = []
        converted_dir = None
        
        for file in data_files:
            if file.suffix == '.urf':
                # 為轉換後的檔案創建專用目錄
                if converted_dir is None:
                    converted_dir = input_dir / 'converted_ohm'
                    converted_dir.mkdir(exist_ok=True)
                    self.logger.info(f"創建轉換檔案目錄: {converted_dir}")
                
                # 轉換URF檔案到專用目錄
                has_terrain = self.config['data']['terrain_file'] is not None
                terrain_file = self.config['data']['terrain_file'] if has_terrain else ''
                
                # 修改 convertURF 以輸出到指定目錄
                original_ohm_path = ERTTimeSeriesProcessor.convertURF(
                    data_read_path=str(file),
                    has_trn=has_terrain,
                    trn_path=terrain_file
                )
                
                # 移動轉換後的檔案到專用目錄
                ohm_filename = Path(original_ohm_path).name
                new_ohm_path = converted_dir / ohm_filename
                
                import shutil
                shutil.move(original_ohm_path, new_ohm_path)
                
                processed_files.append(str(new_ohm_path))
                self.logger.info(f"URF檔案已轉換並移動: {file} -> {new_ohm_path}")
            else:
                processed_files.append(str(file))
        
        # 如果有轉換目錄，更新配置以指向該目錄
        if converted_dir is not None:
            self.converted_dir = str(converted_dir)
        else:
            self.converted_dir = None
        
        return processed_files, file_formats[0]
    
    def create_surveys(self, data_files, file_format):
        """建立測量專案"""
        self.ERT = Project(typ='R2')
        
        if file_format == '.stg':
            if len(data_files) > 1:
                # 若有多個 STG 檔案，使用 Batch 方式載入，才能一次反演全部時段
                input_dir = self.config['data']['input_dir']
                self.ERT.createBatchSurvey(input_dir, ftype='Sting', parser=customStingParser)
                self.logger.info(f"已批次載入 {len(data_files)} 個 STG 檔案 (Batch mode)")
            else:
                # 單一檔案則直接載入
                self.ERT.createSurvey(str(data_files[0]), ftype='Sting', parser=customStingParser)
                self.logger.info(f"已載入STG檔案: {data_files[0]}")
        else:
            # 批次處理 OHM 或 URF(已轉換為OHM) 檔案
            if file_format in ['.ohm', '.urf']:
                ftype = 'BERT'
            else:
                # 對於其他未明確處理的格式，預設為 Sting
                ftype = 'Sting'
            
            if len(data_files) > 1:
                # 多個檔案使用 Batch 方式載入，才能一次反演全部時段
                if file_format == '.urf' and hasattr(self, 'converted_dir') and self.converted_dir:
                    # 對於轉換後的 URF 檔案，使用專用的轉換目錄
                    batch_dir = self.converted_dir
                    self.logger.info(f"使用轉換目錄進行批次載入: {batch_dir}")
                else:
                    # 對於原生 OHM 檔案，使用原始輸入目錄
                    batch_dir = self.config['data']['input_dir']
                
                self.ERT.createBatchSurvey(batch_dir, ftype=ftype)
                self.logger.info(f"已批次載入 {len(data_files)} 個 {ftype} 格式檔案 (Batch mode)")
            else:
                # 單一檔案則直接載入
                self.ERT.createSurvey(data_files[0], ftype=ftype)
                self.logger.info(f"已載入單一 {ftype} 格式檔案: {data_files[0]}")
        
        self.survey_count = len(self.ERT.surveys)
        
        # 輸出資料統計
        for i in range(self.survey_count):
            data_count = len(self.ERT.surveys[i].df)
            self.logger.info(f'測量 {i+1}: 資料點數 {data_count}')
    
    def filter_data(self):
        """資料篩選"""
        self.logger.info("開始資料篩選...")
        
        # 篩選視電阻率
        rho_min = self.config['filter']['rho_min']
        initial_counts = [len(survey.df) for survey in self.ERT.surveys]
        
        self.ERT.filterAppResist(vmin=rho_min)
        
        after_rho_counts = [len(survey.df) for survey in self.ERT.surveys]
        
        for i in range(self.survey_count):
            removed = initial_counts[i] - after_rho_counts[i]
            self.logger.info(f'測量 {i+1}: 移除 {removed} 個負電阻率資料點')
        
        # 檢查是否有互換測量資料
        has_reciprocal = False
        reciprocal_count = 0
        
        for survey in self.ERT.surveys:
            # 檢查 recipError 欄位是否存在且有非 NaN 值
            if 'recipError' in survey.df.columns:
                non_nan_recip_errors = survey.df['recipError'][survey.df['recipError'].notna()]
                if len(non_nan_recip_errors) > 0:
                    has_reciprocal = True
                    reciprocal_count += len(non_nan_recip_errors)
        
        # 如果沒有互換測量，跳過互換誤差篩選
        if not has_reciprocal or reciprocal_count == 0:
            self.logger.info("未找到互換測量資料，跳過互換誤差篩選")
            final_counts = after_rho_counts
        else:
            # 互換誤差篩選
            self.logger.info(f"找到 {reciprocal_count} 個互換測量，進行互換誤差篩選...")
            reciprocal_max = self.config['filter']['reciprocal_error_max']
            
            try:
                self.ERT.filterRecip(percent=reciprocal_max)
                final_counts = [len(survey.df) for survey in self.ERT.surveys]
                
                for i in range(self.survey_count):
                    removed = after_rho_counts[i] - final_counts[i]
                    self.logger.info(f'測量 {i+1}: 移除 {removed} 個高互換誤差資料點')
            except Exception as e:
                self.logger.warning(f"互換誤差篩選失敗: {str(e)}，跳過此步驟")
                final_counts = after_rho_counts
        
        self.logger.info("資料篩選完成")
    
    def plot_pseudo_sections(self):
        """繪製偽剖面"""
        if not self.config['output']['save_pseudo_plots']:
            return
        
        self.logger.info("繪製偽剖面...")
        
        plot_config = self.config['pseudo_plot']
        output_dir = Path(self.config['data']['output_dir']) / 'pseudo_plots'
        
        for i in range(self.survey_count):
            fig, ax = plt.subplots(figsize=plot_config['figsize'])
            
            self.ERT.showPseudo(
                index=i,
                ax=ax,
                cmap=plot_config['cmap'],
                log=plot_config['log_scale'],
                vmin=plot_config['vmin'] if not plot_config['log_scale'] else np.log10(plot_config['vmin']),
                vmax=plot_config['vmax'] if not plot_config['log_scale'] else np.log10(plot_config['vmax'])
            )
            
            ax.set_title(f'Pseudo Section of survey {i}')
            # 儲存圖片
            filename = f'pseudo_section_{i+1:02d}.{plot_config["save_format"]}'
            filepath = output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=plot_config['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'偽剖面已儲存: {filepath}')
    
    def find_common_measurements(self):
        """找出共同的測量點"""
        self.logger.info("尋找共同測量點...")
        
        # 收集所有測量的四極組合
        sets_of_quadruples = []
        
        for i in range(self.survey_count):
            quadruples = set(zip(
                self.ERT.surveys[i].df['a'],
                self.ERT.surveys[i].df['b'],
                self.ERT.surveys[i].df['m'],
                self.ERT.surveys[i].df['n']
            ))
            sets_of_quadruples.append(quadruples)
        
        # 找出交集
        common_quadruples = set.intersection(*sets_of_quadruples)
        self.logger.info(f"找到 {len(common_quadruples)} 個共同測量點")
        
        # 篩選每個測量，只保留共同點
        for i in range(self.survey_count):
            quadruples = list(zip(
                self.ERT.surveys[i].df['a'],
                self.ERT.surveys[i].df['b'],
                self.ERT.surveys[i].df['m'],
                self.ERT.surveys[i].df['n']
            ))
            
            keep_indices = np.array([
                quadruple in common_quadruples for quadruple in quadruples
            ])
            
            self.ERT.surveys[i].filterData(keep_indices)
            
            self.logger.info(f'測量 {i+1}: 保留 {keep_indices.sum()} 個共同測量點')
    
    def fit_error_model(self):
        """擬合誤差模型"""
        self.logger.info("擬合誤差模型...")
        
        # 檢查是否有互換測量資料
        has_reciprocal = False
        reciprocal_count = 0
        
        for survey in self.ERT.surveys:
            # 檢查 recipError 欄位是否存在且有非 NaN 值
            if 'recipError' in survey.df.columns:
                non_nan_recip_errors = survey.df['recipError'][survey.df['recipError'].notna()]
                if len(non_nan_recip_errors) > 0:
                    has_reciprocal = True
                    reciprocal_count += len(non_nan_recip_errors)
        
        if not has_reciprocal or reciprocal_count == 0:
            self.logger.info("未找到互換測量資料，跳過誤差模型擬合")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.ERT.fitErrorLin(ax=ax, index=-1)
                
            if self.config['output']['save_pseudo_plots']:
                output_dir = Path(self.config['data']['output_dir']) / 'pseudo_plots'
                filename = f'error_model.{self.config["pseudo_plot"]["save_format"]}'
                filepath = output_dir / filename
                
                plt.savefig(
                    filepath,
                    dpi=self.config['pseudo_plot']['dpi'],
                    bbox_inches='tight',
                    facecolor='white'
                )
                self.logger.info(f'誤差模型圖已儲存: {filepath}')
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"誤差模型擬合失敗: {str(e)}，跳過此步驟")
            if 'fig' in locals():
                plt.close()
    
    def create_mesh(self):
        """建立網格"""
        self.logger.info("建立反演網格...")
        
        mesh_config = self.config['mesh']
        
        if mesh_config['type'] == 'trian':
            self.ERT.createMesh(
                typ='trian',
                cl=mesh_config['cl'],
                cl_factor=mesh_config['cl_factor'],
                show_output=mesh_config['show_output']
            )
        else:
            self.ERT.createMesh(
                typ='quad',
                elemx=mesh_config['elemx'],
                xgf=mesh_config['xgf'],
                zf=mesh_config['zf'],
                zgf=mesh_config['zgf'],
                pad=mesh_config['pad']
            )
        
        node_count = len(self.ERT.mesh.node)
        cell_count = len(self.ERT.mesh.cell_type)
        
        self.logger.info(f'網格建立完成 - 節點數: {node_count}, 元素數: {cell_count}')
        
        # 繪製網格
        if self.config['output']['save_mesh_plot']:
            fig, ax = plt.subplots(figsize=(12, 8))
            self.ERT.showMesh(ax=ax)
                        
            output_dir = Path(self.config['data']['output_dir']) / 'mesh_plots'
            filename = f'mesh.{self.config["result_plot"]["save_format"]}'
            filepath = output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=self.config['result_plot']['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'網格圖已儲存: {filepath}')
    
    def run_inversion(self):
        """執行反演"""
        self.logger.info("開始反演計算...")
        
        inversion_config = self.config['inversion']
        
        # 設置反演參數
        self.ERT.param['tolerance'] = inversion_config['tolerance']
        
        # 執行初始反演
        self.ERT.invert(
            iplot=False,
        )
        
        self.logger.info("初始反演完成")
        self.logger.info(self.ERT.summary())
        
        # 繪製收斂曲線
        if self.config['output']['save_convergence_plot']:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.ERT.showRMS(ax=ax)
                        
            output_dir = Path(self.config['data']['output_dir']) / 'convergence_plots'
            filename = f'convergence.{self.config["result_plot"]["save_format"]}'
            filepath = output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=self.config['result_plot']['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'收斂曲線已儲存: {filepath}')
        
        # 迭代移除離群值
        if inversion_config['remove_outliers']:
            self._iterative_inversion(inversion_config)
    
    def _iterative_inversion(self, inversion_config):
        """迭代反演，移除離群值"""
        max_iterations = inversion_config['max_iterations']
        outlier_threshold = inversion_config['outlier_threshold']
        iteration = 0

        
        self.logger.info("開始迭代反演，移除離群值...")
        
        while (self.ERT.invLog.count('Solution converged') < self.ERT.pinfo['Number of Surveys'] and 
               iteration < max_iterations):
            
            self.logger.info(f"第 {iteration + 1} 次迭代 - 移除離群值")
            
            # 移除擬合度最差的資料
            for i in range(self.survey_count):
                
                # 若缺少 resInvError 欄位則跳過該測量
                if 'resInvError' not in self.ERT.surveys[i].df.columns:
                    self.logger.warning(f"測量 {i+1}: 找不到 resInvError 欄位，跳過離群值移除")
                    continue

                misfit = np.abs(self.ERT.surveys[i].df['resInvError'])
                threshold_index = int(np.round((1 - outlier_threshold) * len(misfit)))
                rm_target = np.argsort(misfit)[threshold_index:]
                
                keep_indices = np.ones(len(misfit), dtype=bool)
                keep_indices[rm_target] = False
                
                removed_count = len(rm_target)
                self.ERT.surveys[i].filterData(keep_indices)
                
                self.logger.info(f'測量 {i+1}: 移除 {removed_count} 個離群值')
            
            # 重新反演
            self.ERT.invert(iplot=False,)
            # 更新誤差欄位供下一輪迭代使用
            try:
                self.ERT.getInvError()
            except Exception as e:
                self.logger.warning(f"迭代 {iteration+1} 載入反演誤差資料失敗: {str(e)}")
            iteration += 1
        
        # 檢查收斂狀態
        converged_count = self.ERT.invLog.count('Solution converged')
        if converged_count == self.survey_count:
            self.logger.info("所有反演已收斂")
        else:
            self.logger.warning(f"迭代結束，{converged_count}/{self.survey_count} 個反演收斂")
    
    def plot_results(self):
        """繪製反演結果"""
        if not self.config['output']['save_result_plots']:
            return
        
        self.logger.info("繪製反演結果...")
        
        # 取得可用的反演結果數量
        available_results = len(self.ERT.meshResults) if hasattr(self.ERT, 'meshResults') else 0

        if available_results == 0:
            self.logger.warning("未找到任何反演結果 (meshResults 為空)，跳過結果繪圖。")
            return

        result_config = self.config['result_plot']
        output_dir = Path(self.config['data']['output_dir']) / 'result_plots'
        
        # 計算色階範圍
        vmin_config = result_config['vmin']
        vmax_config = result_config['vmax'] 
        log_scale = result_config['log_scale']
        
        if log_scale:
            vmin_display = np.log10(vmin_config)
            vmax_display = np.log10(vmax_config)
        else:
            vmin_display = vmin_config
            vmax_display = vmax_config
        
        for i in range(available_results):
            fig, ax = plt.subplots(figsize=result_config['figsize'])
            
            self.ERT.showResults(
                index=i,
                ax=ax,
                attr='Resistivity(log10)',
                sens=result_config['show_sensitivity'],
                contour=result_config['contour'],
                vmin=vmin_display,
                vmax=vmax_display,
                color_map=result_config['color_map'],
                clipCorners=result_config['clip_corners'],
                maxDepth=result_config['max_depth']
            )

            # 強制設置色階條範圍
            mesh_result = self.ERT.meshResults[i]
            if hasattr(mesh_result, 'cbar') and mesh_result.cbar is not None:
                # 設置色階條範圍
                mesh_result.cbar.mappable.set_clim(vmin=vmin_display, vmax=vmax_display)
                
                # 設置刻度標籤（如果是對數尺度）
                if log_scale:
                    # 創建均勻分布的刻度
                    n_ticks = 11  # 刻度數量
                    tick_positions = np.linspace(vmin_display, vmax_display, n_ticks)
                    tick_labels = [f'{10**pos:.0f}' if pos >= 2 else f'{10**pos:.1f}' 
                                 for pos in tick_positions]
                    mesh_result.cbar.set_ticks(tick_positions)
                    mesh_result.cbar.set_ticklabels(tick_labels)
                
                self.logger.info(f'色階條範圍已設置為: {vmin_display:.3f} - {vmax_display:.3f}')

            # 如果是 contour=True 且需要加強角落裁切，手動加強裁切效果
            if result_config['contour'] and result_config['clip_corners']:
                self._enhance_corner_clipping(ax, i)
            
            # 設置X軸範圍
            ax.set_xlim([self.ERT.elec['x'].min(), self.ERT.elec['x'].max()])
            ax.set_title(f'Survey {i}')
            
            # 儲存結果圖
            filename = f'resistivity_{i+1:02d}.{result_config["save_format"]}'
            filepath = output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=result_config['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'反演結果已儲存: {filepath}')
        
        # 繪製反演誤差
        if self.config['output']['save_error_plots']:
            self._plot_inversion_errors()
    
    def _enhance_corner_clipping(self, ax, index):
        """加強角落裁切效果（針對 contour=True 的情況）"""
        try:
            # 獲取電極位置
            elec_x = self.ERT.elec['x'].values
            elec_z = self.ERT.elec['z'].values
            elec_xmin, elec_xmax = np.min(elec_x), np.max(elec_x)
            
            # 創建更嚴格的角落遮罩
            from matplotlib.patches import Polygon
            import matplotlib.patches as patches
            
            # 計算角落裁切的幾何形狀（基於 ResIPy 的邏輯）
            if hasattr(self.ERT, 'trapeziod') and self.ERT.trapeziod is not None:
                # 使用 ResIPy 計算的梯形遮罩
                trap_vertices = self.ERT.trapeziod
                
                # 為每個等值線集合添加更嚴格的裁切
                for collection in ax.collections:
                    if hasattr(collection, 'set_clip_path'):
                        # 創建裁切路徑
                        clip_path = patches.Polygon(trap_vertices, 
                                                facecolor='none', 
                                                edgecolor='none',
                                                transform=ax.transData)
                        ax.add_patch(clip_path)
                        collection.set_clip_path(clip_path)
            
        except Exception as e:
            self.logger.warning(f"加強角落裁切失敗: {str(e)}")
            
    def _plot_inversion_errors(self):
        """繪製反演誤差"""
        self.logger.info("繪製反演誤差...")
        
        error_output_dir = Path(self.config['data']['output_dir']) / 'error_plots'
        result_config = self.config['result_plot']
        
        for i in range(self.survey_count):
            # 若沒有 resInvError 欄位則跳過
            if 'resInvError' not in self.ERT.surveys[i].df.columns:
                self.logger.warning(f"測量 {i+1}: 缺少 resInvError 資料，跳過誤差圖繪製")
                continue
            fig, ax = plt.subplots(figsize=result_config['figsize'])
            
            self.ERT.showInvError(index=i, ax=ax)
                        
            filename = f'inversion_error_{i+1:02d}.{result_config["save_format"]}'
            filepath = error_output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=result_config['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'反演誤差圖已儲存: {filepath}')
    
    def save_numerical_data(self):
        """儲存數值資料"""
        if not self.config['output']['save_numerical_data']:
            return
        
        self.logger.info("儲存數值資料...")
        
        output_dir = Path(self.config['data']['output_dir']) / 'numerical_data'
        data_format = self.config['output']['data_format']
        
        # 建立 VTK 輸出目錄
        vtk_output_dir = output_dir / 'vtk_results'
        vtk_output_dir.mkdir(exist_ok=True)
        
        # 1. 輸出 VTK 格式（使用 ResIPy 內建功能）
        try:
            self.logger.info("輸出 VTK 格式反演結果...")
            
            # 使用 saveVtks 輸出所有反演結果為 VTK 格式
            self.ERT.saveVtks(dirname=str(vtk_output_dir))
            self.logger.info(f"VTK 檔案已儲存至: {vtk_output_dir}")
            
            # 可選：使用 exportMeshResults 輸出更多格式
            # self.ERT.exportMeshResults(
            #     dirname=str(vtk_output_dir), 
            #     ftype='vtk',
            #     voxel=False
            # )
            
        except Exception as e:
            self.logger.error(f"VTK 輸出失敗: {str(e)}")
        
        # 2. 輸出 CSV 格式（保持原有功能）
        self.logger.info("輸出 CSV 格式數值資料...")
        
        for i in range(self.survey_count):
            try:
                # 取得電阻率結果 - 使用 meshResults
                if hasattr(self.ERT, 'meshResults') and len(self.ERT.meshResults) > i:
                    mesh_result = self.ERT.meshResults[i].df
                    
                    # 嘗試不同的電阻率欄位名稱
                    resistivity_columns = [
                        'Resistivity(ohm.m)', 
                        'Resistivity(Ohm-m)', 
                        'Resistivity',
                        'Magnitude(ohm.m)'
                    ]
                    
                    resistivity = None
                    for col_name in resistivity_columns:
                        if col_name in mesh_result.columns:
                            resistivity = mesh_result[col_name].values
                            self.logger.info(f"測量 {i+1}: 使用電阻率欄位 '{col_name}'")
                            break
                    
                    if resistivity is None:
                        self.logger.warning(f"測量 {i+1}: 找不到電阻率資料欄位，可用欄位: {list(mesh_result.columns)}")
                        continue
                    
                    # 獲取座標資訊
                    if 'X' in mesh_result.columns and 'Z' in mesh_result.columns:
                        x_coords = mesh_result['X'].values
                        z_coords = mesh_result['Z'].values  # ResIPy中Z通常代表深度/高程
                    elif 'x' in mesh_result.columns and 'z' in mesh_result.columns:
                        x_coords = mesh_result['x'].values
                        z_coords = mesh_result['z'].values
                    else:
                        self.logger.warning(f"測量 {i+1}: 找不到座標資料，使用網格中心點")
                        x_coords = self.ERT.mesh.elm_center[:, 0]
                        z_coords = self.ERT.mesh.elm_center[:, 1]
                    
                else:
                    self.logger.warning(f"測量 {i+1}: meshResults 不存在，嘗試其他方法獲取電阻率結果")
                    continue
                                    
                # 建立資料框架
                data_dict = {
                    'x': x_coords,
                    'z': z_coords,
                    'resistivity': resistivity
                }
                
                df = pd.DataFrame(data_dict)
                
                # 儲存 CSV 資料
                if data_format.lower() == 'csv':
                    filename = f'resistivity_data_{i+1:02d}.csv'
                    filepath = output_dir / filename
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                else:
                    filename = f'resistivity_data_{i+1:02d}.txt'
                    filepath = output_dir / filename
                    df.to_csv(filepath, index=False, sep='\t', encoding='utf-8')
                
                self.logger.info(f'CSV 數值資料已儲存: {filepath} (共 {len(df)} 個資料點)')
                
            except Exception as e:
                self.logger.error(f"儲存測量 {i+1} 的 CSV 數值資料時發生錯誤: {str(e)}")
                continue
        
        # 3. 輸出摘要資訊
        self.logger.info("=" * 50)
        self.logger.info("數值資料輸出摘要:")
        self.logger.info(f"• CSV 格式: {output_dir}")
        self.logger.info(f"• VTK 格式: {vtk_output_dir}")
        self.logger.info(f"• 處理測量數: {self.survey_count}")
        self.logger.info("=" * 50)
        
        # 儲存處理摘要
        self._save_processing_summary()
    
    def _save_processing_summary(self):
        """儲存處理摘要"""
        output_dir = Path(self.config['data']['output_dir']) / 'numerical_data'
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'survey_count': self.survey_count,
            'total_nodes': len(self.ERT.mesh.node) if self.ERT.mesh else 0,
            'total_elements': len(self.ERT.mesh.cell_type) if self.ERT.mesh else 0,
            'convergence_status': self.ERT.invLog.count('Solution converged'),
            # 若缺少 resInvError 欄位則設為 None
            'final_rms': (self.ERT.surveys[-1].df['resInvError'].abs().mean() \
                          if self.ERT.surveys and 'resInvError' in self.ERT.surveys[-1].df.columns else None),
            'config_used': self.config
        }
        
        # 儲存為YAML格式
        summary_file = output_dir / 'processing_summary.yaml'
        with open(summary_file, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f'處理摘要已儲存: {summary_file}')
    
    def run_complete_processing(self):
        """執行完整的處理流程"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("開始時序ERT資料處理")
            self.logger.info("=" * 60)
            
            # 1. 載入資料
            data_files, file_format = self.load_data()
            
            # 2. 建立測量專案
            self.create_surveys(data_files, file_format)
            
            # 3. 資料篩選
            self.filter_data()
            
            # 4. 繪製偽剖面
            self.plot_pseudo_sections()
            
            # 5. 找出共同測量點
            self.find_common_measurements()
            
            # 6. 擬合誤差模型
            self.fit_error_model()
            
            # 7. 建立網格
            self.create_mesh()
            
            # 8. 執行反演
            self.run_inversion()
            
            # 9. 繪製結果
            self.plot_results()
            
            # 10. 儲存數值資料
            self.save_numerical_data()
            
            self.logger.info("=" * 60)
            self.logger.info("時序ERT資料處理完成")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"處理過程發生錯誤: {str(e)}")
            raise

def create_sample_data(output_dir):
    """創建範例資料集結構"""
    data_dir = Path(output_dir) / 'sample_data'
    data_dir.mkdir(exist_ok=True)
    
    # 創建範例配置檔案
    sample_config = {
        'data': {
            'input_dir': str(data_dir),
            'output_dir': 'output',
            'supported_formats': ['.stg', '.ohm', '.urf'],
            'terrain_file': None
        },
        'filter': {
            'rho_min': 0.0,
            'reciprocal_error_max': 20.0
        },
        'pseudo_plot': {
            'figsize': [12, 8],
            'cmap': 'jet',
            'log_scale': True,
            'vmin': 32,
            'vmax': 3162,
            'dpi': 300,
            'save_format': 'png'
        },
        'mesh': {
            'type': 'trian',
            'cl': 0.75,
            'cl_factor': 5,
            'show_output': False
        },
        'inversion': {
            'tolerance': 5,
            'max_iterations': 10,
            'parallel': False,
            'remove_outliers': True,
            'outlier_threshold': 0.05
        },
        'result_plot': {
            'figsize': [12, 8],
            'color_map': 'jet',
            'contour': True,
            'clip_corners': True,
            'max_depth': 100,
            'show_sensitivity': False,
            'dpi': 300,
            'save_format': 'png'
        },
        'output': {
            'save_pseudo_plots': True,
            'save_mesh_plot': True,
            'save_result_plots': True,
            'save_error_plots': True,
            'save_convergence_plot': True,
            'save_numerical_data': True,
            'data_format': 'csv'
        },
        'time_series': {
            'create_animation': True,
            'animation_format': 'gif',
            'frame_duration': 1.0,
            'difference_analysis': True,
            'reference_index': 0
        },
        'logging': {
            'level': 'INFO',
            'save_log': True,
            'log_file': 'processing.log'
        }
    }
    
    config_file = Path(output_dir) / 'sample_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"範例配置檔案已創建: {config_file}")
    print(f"範例資料目錄已創建: {data_dir}")
    print("\n請將您的STG資料檔案複製到範例資料目錄中")
    print("然後執行: python run_ert_analysis.py --config sample_config.yaml")


def validate_config(config_file):
    """驗證配置檔案"""
    if not Path(config_file).exists():
        raise FileNotFoundError(f"配置檔案不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 檢查必要的配置項目
    required_sections = ['data', 'filter', 'mesh', 'inversion', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置檔案缺少必要的區段: {section}")
    
    # 檢查輸入目錄是否存在
    input_dir = Path(config['data']['input_dir'])
    if not input_dir.exists():
        raise FileNotFoundError(f"輸入資料目錄不存在: {input_dir}")
    
    print(f"配置檔案驗證通過: {config_file}")
    return config


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='ERT時序監測資料處理系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 創建範例配置和目錄結構
  python run_ert_analysis.py --create-sample ./sample_project
  
  # 執行完整分析
  python run_ert_analysis.py --config config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置檔案路徑 (預設: config.yaml)'
    )
    
    parser.add_argument(
        '--create-sample',
        type=str,
        metavar='DIR',
        help='創建範例專案結構到指定目錄'
    )
        
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='顯示詳細輸出'
    )
    
    args = parser.parse_args()
    
    try:
        # 創建範例專案
        if args.create_sample:
            create_sample_data(args.create_sample)
            return
        
        # 驗證配置檔案
        config = validate_config(args.config)
        
        # 調整日誌等級
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
        
        print("=" * 80)
        print("ERT時序監測資料處理系統")
        print("Time-lapse ERT Data Processing System")
        print("=" * 80)
        print(f"配置檔案: {args.config}")
        print(f"輸入目錄: {config['data']['input_dir']}")
        print(f"輸出目錄: {config['data']['output_dir']}")
        print("-" * 80)
        
        # 建立ERT處理器
        processor = ERTTimeSeriesProcessor(args.config)
        
        # 執行ERT資料處理
        print("開始ERT資料處理...")
        processor.run_complete_processing()
        print("ERT資料處理完成!")
        
        print("\n" + "=" * 80)
        print("所有處理完成!")
        print(f"結果已儲存至: {config['data']['output_dir']}")
        print("=" * 80)
        
        # 顯示輸出摘要
        output_dir = Path(config['data']['output_dir'])
        print("\n輸出文件摘要:")
        
        subdirs = ['pseudo_plots', 'mesh_plots', 'result_plots', 
                   'error_plots', 'convergence_plots', 'numerical_data']
        
        for subdir in subdirs:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob('*')))
                print(f"  {subdir}: {file_count} 個檔案")
        
        # 檢查日誌檔案
        log_file = Path(config['logging']['log_file'])
        if log_file.exists():
            print(f"  處理日誌: {log_file}")
        
    except KeyboardInterrupt:
        print("\n\n使用者中斷程式執行")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 