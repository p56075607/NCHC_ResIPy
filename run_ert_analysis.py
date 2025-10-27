#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERT時序監測資料處理主程式
Main Program for Time-lapse ERT Data Processing

功能包含：
- 完整的ERT資料處理流程
- 時序分析與動畫製作
- 自動化批次處理
- 綜合報告生成

作者: CHEN CHUN
日期: 2025
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

# 添加當前目錄到Python路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ert_time_series_processor import ERTTimeSeriesProcessor

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