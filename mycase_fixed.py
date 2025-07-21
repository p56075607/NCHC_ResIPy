# %%
import os
import pickle
from resipy import Project
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from custom_parser import customStingParser  # 導入自定義解析器
# %%

data_dir = r'C:\Users\Git\resipy\jupyter-notebook\mycase\dataset_stg\urf_two\converted_ohm'
file_format = [f[-4:] for f in os.listdir(data_dir)]
# Check file format: only accept .ohm .stg and (.urf)

ERT = Project(typ='R2') # create a Project object in a working directory
if len(file_format) >= 1 and all(x == file_format[0] for x in file_format): 
    # if all same format, Use the batch funtion
    if file_format[0] == '.stg':
        # 使用自定義解析器來處理 .stg 文件
        for f in os.listdir(data_dir):
            if f.endswith('.stg'):
                file_path = os.path.join(data_dir, f)
                # 使用自定義解析器
                ERT.createSurvey(file_path, ftype='Sting', parser=customStingParser)
    else:
        ftype = 'BERT' if file_format[0] == '.ohm' else 'Sting'
        ERT.createBatchSurvey(os.path.join(data_dir), ftype=ftype)
    
    for i in range(len(ERT.surveys)):
        print('Survey {:d}, data numbers: {:d}'.format(i,len(ERT.surveys[i].df)) )
    ERT.filterAppResist(vmin=0)
    for i in range(len(ERT.surveys)):
        print('Survey {:d}, data numbers: {:d}'.format(i,len(ERT.surveys[i].df)) )
    ERT.filterRecip(percent=20) # in this case this only removes one quadrupoles with reciprocal error bigger than 20 percent

    # plot rhoa pseudo-section
    for i in range(len(ERT.surveys)):
        fig, ax = plt.subplots(figsize=(10,5))
        ERT.showPseudo(index = i, ax = ax, cmap='jet',log=True)

    # Fit error model
    # 取交集才能一起畫
    # Four electrode numbers for each data set
    sets_of_quadruples = []

    for i in range(len(ERT.surveys)):
        quadruples = set(zip(ERT.surveys[i].df['a'], ERT.surveys[i].df['b'], ERT.surveys[i].df['m'], ERT.surveys[i].df['n']))
        sets_of_quadruples.append(quadruples)

    # find the common quadruples
    common_quadruples = set.intersection(*sets_of_quadruples)
    for i in range(len(ERT.surveys)):
        quadruples = list(zip(ERT.surveys[i].df['a'], ERT.surveys[i].df['b'], ERT.surveys[i].df['m'], ERT.surveys[i].df['n']))
        keep_indices = np.array([quadruple in common_quadruples for quadruple in quadruples])
        ERT.surveys[i].filterData(keep_indices)
        print(ERT.surveys[i])

    fig, ax = plt.subplots()
    ERT.fitErrorLin(ax=ax,index=-1)
    # %%
    # Creat mesh
    ERT.createMesh(typ='trian',cl=0.75,cl_factor=5, show_output=False) # this actually call gmsh.exe to create the mesh
    # ERT.createMesh(typ='quad', elemx=2, xgf=1.5, zf=1.1, zgf=1.5, pad=2)
    ERT.showMesh()
    print('Total node numbers: ',len(ERT.mesh.node))
    print('Total cell numbers: ',len(ERT.mesh.cell_type))

    # %%Inversion
    ERT.param['tolerance'] = 5
    ERT.invert(iplot=False,parallel=True)
    print(ERT.summary())
    # Plot convergence curve
    ERT.showRMS()
    for i in range(len(ERT.surveys)):
        # Plot result
        fig, ax = plt.subplots(figsize=(10,5))
        ERT.showResults(index = i,ax = ax, attr='Resistivity(log10)', sens=False, contour=True, 
                        vmin=np.log10(32), vmax=np.log10(3162), 
                        color_map='jet',
                        clipCorners=True,
                        maxDepth=100,
                        )
        ax.set_xlim([ERT.elec['x'].min(),ERT.elec['x'].max()])        
        # Plot inverison error 
        ERT.showInvError(index = i, )

    # Initialize variables and set the maximum number of iterations to avoid infinite loops
    max_iterations = 10
    iteration = 0

    # Perform inversion and repeat data removal and inversion if convergence is not achieved
    while ERT.invLog.count('Solution converged') < len(ERT.surveys) and iteration < max_iterations:
        print(f"第 {iteration + 1} 次反演未完全收斂，進行資料剔除並重複反演")
        
        # 移除擬合度最差的 5% 資料
        for i in range(len(ERT.surveys)):
            misfit = np.abs(ERT.surveys[i].df['resInvError'])
            rm_target = np.argsort(misfit)[int(np.round(0.95 * len(misfit))):]
            keep_indx = np.ones(len(misfit), dtype=bool)
            keep_indx[rm_target] = False
            ERT.surveys[i].filterData(keep_indx)
            ERT.showInvError(index=i)
        
        # Perform the inversion and plot the results
        ERT.invert(iplot=True)
        
        # Update iteration count
        iteration += 1

    # 根據最終收斂狀態輸出結果
    if ERT.invLog.count('Solution converged') == len(ERT.surveys):
        print("所有反演已收斂，無需進一步資料剔除")
    else:
        print("達到最大迭代次數，但仍有部分反演未收斂，請檢查資料或模型參數")
# %%
else: # show error: please check the data direction!
    raise ValueError('please check the data direction and format!') 