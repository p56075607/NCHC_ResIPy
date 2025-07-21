cd /d "%~dp0"
call C:\Users\B30122\anaconda3\Scripts\activate.bat resipy
call python run_ert_analysis.py --config config.yaml
pause