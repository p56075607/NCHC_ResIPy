#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERT時序分析工具
Time-lapse ERT Analysis Tool

功能包含：
- 時序剖面動畫製作
- 差值分析
- 變化量統計
- 時序趨勢分析

作者: AI Assistant
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from pathlib import Path
import yaml
import logging


class TimeSeriesAnalyzer:
    """ERT時序分析器"""
    
    def __init__(self, ert_processor, config):
        """
        初始化時序分析器
        
        參數:
            ert_processor: ERT處理器實例
            config: 配置參數
        """
        self.ert = ert_processor.ERT
        self.config = config
        self.survey_count = ert_processor.survey_count
        self.logger = ert_processor.logger
        
        # 建立時序分析輸出目錄
        self.output_dir = Path(config['data']['output_dir']) / 'time_series_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
    def create_animation(self):
        """創建時序剖面動畫"""
        if not self.config['time_series']['create_animation']:
            return
        
        self.logger.info("創建時序剖面動畫...")
        
        # 設置動畫參數
        fig, ax = plt.subplots(figsize=self.config['result_plot']['figsize'])
        
        # 初始化第一幀
        im = self.ert.showResults(
            index=0,
            ax=ax,
            attr='Resistivity(log10)',
            contour=self.config['result_plot']['contour'],
            vmin=np.log10(self.config['pseudo_plot']['vmin']),
            vmax=np.log10(self.config['pseudo_plot']['vmax']),
            color_map=self.config['result_plot']['color_map'],
            clipCorners=self.config['result_plot']['clip_corners'],
            maxDepth=self.config['result_plot']['max_depth']
        )
        
        ax.set_xlim([self.ert.elec['x'].min(), self.ert.elec['x'].max()])
        ax.set_xlabel('距離 (m)', fontsize=12)
        ax.set_ylabel('深度 (m)', fontsize=12)
        
        # 添加時間標籤
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=14, fontweight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="white", alpha=0.8))
        
        def animate(frame):
            """動畫更新函數"""
            ax.clear()
            
            # 繪製當前幀的電阻率分佈
            self.ert.showResults(
                index=frame,
                ax=ax,
                attr='Resistivity(log10)',
                contour=self.config['result_plot']['contour'],
                vmin=np.log10(self.config['pseudo_plot']['vmin']),
                vmax=np.log10(self.config['pseudo_plot']['vmax']),
                color_map=self.config['result_plot']['color_map'],
                clipCorners=self.config['result_plot']['clip_corners'],
                maxDepth=self.config['result_plot']['max_depth']
            )
            
            ax.set_xlim([self.ert.elec['x'].min(), self.ert.elec['x'].max()])
            ax.set_title(f'時序電阻率剖面 - 測量 {frame+1}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('距離 (m)', fontsize=12)
            ax.set_ylabel('深度 (m)', fontsize=12)
            
            # 更新時間標籤
            time_text = ax.text(0.02, 0.98, f'測量 {frame+1}/{self.survey_count}', 
                               transform=ax.transAxes, 
                               fontsize=14, fontweight='bold',
                               verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor="white", alpha=0.8))
            
            return [time_text]
        
        # 創建動畫
        anim = FuncAnimation(
            fig, animate, frames=self.survey_count,
            interval=self.config['time_series']['frame_duration'] * 1000,
            blit=False, repeat=True
        )
        
        # 儲存動畫
        animation_format = self.config['time_series']['animation_format']
        filename = f'resistivity_time_series.{animation_format}'
        filepath = self.output_dir / filename
        
        if animation_format.lower() == 'gif':
            writer = PillowWriter(fps=1/self.config['time_series']['frame_duration'])
            anim.save(filepath, writer=writer)
        else:
            anim.save(filepath, writer='ffmpeg', fps=1/self.config['time_series']['frame_duration'])
        
        plt.close()
        self.logger.info(f'時序動畫已儲存: {filepath}')
        
    def create_difference_analysis(self):
        """創建差值分析"""
        if not self.config['time_series']['difference_analysis']:
            return
        
        self.logger.info("創建差值分析...")
        
        reference_index = self.config['time_series']['reference_index']
        
        # 取得參考電阻率
        ref_resistivity = self.ert.surveys[reference_index].res
        
        # 計算每個時間點與參考的差值
        for i in range(self.survey_count):
            if i == reference_index:
                continue
                
            current_resistivity = self.ert.surveys[i].res
            
            # 計算對數差值（百分比變化）
            log_diff = np.log10(current_resistivity) - np.log10(ref_resistivity)
            
            # 繪製差值分佈
            fig, ax = plt.subplots(figsize=self.config['result_plot']['figsize'])
            
            # 使用現有的網格結構繪製差值
            mesh = self.ert.mesh
            x_centers = mesh.elm_center[:, 0]
            z_centers = mesh.elm_center[:, 1]
            
            # 創建scatter plot
            scatter = ax.scatter(
                x_centers, z_centers, 
                c=log_diff, 
                cmap='RdBu_r',  # 紅藍配色，紅色表示增加，藍色表示減少
                vmin=-0.5, vmax=0.5,  # 限制在±50%變化範圍
                s=20, alpha=0.7
            )
            
            # 添加色彩條
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('電阻率變化 (log10比值)', fontsize=12)
            
            ax.set_xlim([self.ert.elec['x'].min(), self.ert.elec['x'].max()])
            ax.set_ylim([z_centers.min(), 0])  # 限制深度顯示
            ax.set_title(f'電阻率變化 - 測量{i+1} vs 測量{reference_index+1}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('距離 (m)', fontsize=12)
            ax.set_ylabel('深度 (m)', fontsize=12)
            ax.invert_yaxis()  # 反轉Y軸使深度向下
            
            # 儲存差值圖
            filename = f'difference_{i+1:02d}_vs_{reference_index+1:02d}.png'
            filepath = self.output_dir / filename
            
            plt.savefig(
                filepath,
                dpi=self.config['result_plot']['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            self.logger.info(f'差值分析圖已儲存: {filepath}')
        
        # 創建變化量統計
        self._create_change_statistics(ref_resistivity, reference_index)
    
    def _create_change_statistics(self, ref_resistivity, reference_index):
        """創建變化量統計"""
        self.logger.info("創建變化量統計...")
        
        # 儲存統計資料
        stats_data = []
        
        for i in range(self.survey_count):
            if i == reference_index:
                stats_data.append({
                    'survey': i + 1,
                    'mean_resistivity': np.mean(ref_resistivity),
                    'std_resistivity': np.std(ref_resistivity),
                    'mean_change_percent': 0.0,
                    'std_change_percent': 0.0,
                    'max_increase_percent': 0.0,
                    'max_decrease_percent': 0.0
                })
                continue
            
            current_resistivity = self.ert.surveys[i].res
            
            # 計算變化百分比
            change_percent = ((current_resistivity - ref_resistivity) / ref_resistivity) * 100
            
            stats_data.append({
                'survey': i + 1,
                'mean_resistivity': np.mean(current_resistivity),
                'std_resistivity': np.std(current_resistivity),
                'mean_change_percent': np.mean(change_percent),
                'std_change_percent': np.std(change_percent),
                'max_increase_percent': np.max(change_percent),
                'max_decrease_percent': np.min(change_percent)
            })
        
        # 建立統計資料框架
        stats_df = pd.DataFrame(stats_data)
        
        # 儲存統計資料
        stats_file = self.output_dir / 'change_statistics.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        self.logger.info(f'變化量統計已儲存: {stats_file}')
        
        # 繪製統計圖表
        self._plot_change_trends(stats_df)
    
    def _plot_change_trends(self, stats_df):
        """繪製變化趨勢圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 平均電阻率趨勢
        axes[0, 0].plot(stats_df['survey'], stats_df['mean_resistivity'], 
                       'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('平均電阻率趨勢', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('測量次數', fontsize=10)
        axes[0, 0].set_ylabel('平均電阻率 (Ω·m)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均變化百分比趨勢
        axes[0, 1].plot(stats_df['survey'], stats_df['mean_change_percent'], 
                       'ro-', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('平均變化百分比趨勢', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('測量次數', fontsize=10)
        axes[0, 1].set_ylabel('變化百分比 (%)', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最大增加/減少趨勢
        axes[1, 0].plot(stats_df['survey'], stats_df['max_increase_percent'], 
                       'g^-', linewidth=2, markersize=6, label='最大增加')
        axes[1, 0].plot(stats_df['survey'], stats_df['max_decrease_percent'], 
                       'rv-', linewidth=2, markersize=6, label='最大減少')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('極值變化趨勢', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('測量次數', fontsize=10)
        axes[1, 0].set_ylabel('變化百分比 (%)', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 標準差趨勢
        axes[1, 1].plot(stats_df['survey'], stats_df['std_resistivity'], 
                       'mo-', linewidth=2, markersize=6)
        axes[1, 1].set_title('電阻率標準差趨勢', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('測量次數', fontsize=10)
        axes[1, 1].set_ylabel('標準差 (Ω·m)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 儲存趨勢圖
        trends_file = self.output_dir / 'change_trends.png'
        plt.savefig(
            trends_file,
            dpi=self.config['result_plot']['dpi'],
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        
        self.logger.info(f'變化趨勢圖已儲存: {trends_file}')
    
    def create_comprehensive_report(self):
        """創建綜合分析報告"""
        self.logger.info("創建綜合分析報告...")
        
        # 讀取統計資料
        stats_file = self.output_dir / 'change_statistics.csv'
        if not stats_file.exists():
            self.logger.warning("統計資料不存在，跳過報告生成")
            return
        
        stats_df = pd.read_csv(stats_file)
        
        # 生成報告內容
        report_content = f"""
# ERT時序分析報告
## Time-lapse ERT Analysis Report

### 基本資訊
- 測量次數: {self.survey_count}
- 處理日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 參考測量: 第{self.config['time_series']['reference_index'] + 1}次

### 電阻率變化摘要
- 最大平均增加: {stats_df['mean_change_percent'].max():.2f}%
- 最大平均減少: {stats_df['mean_change_percent'].min():.2f}%
- 變化標準差: {stats_df['mean_change_percent'].std():.2f}%

### 極值變化
- 單點最大增加: {stats_df['max_increase_percent'].max():.2f}%
- 單點最大減少: {stats_df['max_decrease_percent'].min():.2f}%

### 各測量統計

| 測量 | 平均電阻率 | 標準差 | 平均變化% | 最大增加% | 最大減少% |
|------|------------|--------|-----------|-----------|-----------|
"""
        
        for _, row in stats_df.iterrows():
            report_content += f"| {int(row['survey'])} | {row['mean_resistivity']:.2f} | {row['std_resistivity']:.2f} | {row['mean_change_percent']:.2f} | {row['max_increase_percent']:.2f} | {row['max_decrease_percent']:.2f} |\n"
        
        report_content += f"""
### 分析結論
1. 電阻率時序變化{'穩定' if stats_df['mean_change_percent'].std() < 10 else '明顯'}
2. {'檢測到顯著的電阻率增加趨勢' if stats_df['mean_change_percent'].iloc[-1] > 20 else '電阻率變化在正常範圍內'}
3. {'存在局部異常區域' if stats_df['max_increase_percent'].max() > 100 or stats_df['max_decrease_percent'].min() < -50 else '整體變化均勻'}

### 文件說明
- `resistivity_time_series.gif`: 時序動畫
- `difference_*.png`: 各時間點差值分佈圖
- `change_statistics.csv`: 詳細統計資料
- `change_trends.png`: 變化趨勢圖表

---
*此報告由ERT時序分析系統自動生成*
"""
        
        # 儲存報告
        report_file = self.output_dir / 'analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f'綜合分析報告已儲存: {report_file}')
    
    def run_complete_analysis(self):
        """執行完整的時序分析"""
        try:
            self.logger.info("開始時序分析...")
            
            # 1. 創建動畫
            self.create_animation()
            
            # 2. 差值分析
            self.create_difference_analysis()
            
            # 3. 生成綜合報告
            self.create_comprehensive_report()
            
            self.logger.info("時序分析完成")
            
        except Exception as e:
            self.logger.error(f"時序分析過程發生錯誤: {str(e)}")
            raise 