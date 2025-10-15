cd /d "%~dp0"
call C:\ProgramData\anaconda3\Scripts\activate.bat resipy
call python run_ert_analysis.py --config config.yaml
pause