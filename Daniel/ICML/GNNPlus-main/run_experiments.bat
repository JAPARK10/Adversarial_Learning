@echo off
echo Starting Standard Experiment (150 epochs)...
python main.py --cfg configs/gcn/rfid_lopo_windowed_3_standard.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Standard experiment failed!
    exit /b %ERRORLEVEL%
)
echo Standard Experiment Completed.

echo Starting Tuned Experiment (80 epochs)...
python main.py --cfg configs/gcn/rfid_lopo_windowed_3_tuned.yaml
if %ERRORLEVEL% NEQ 0 (
    echo Tuned experiment failed!
    exit /b %ERRORLEVEL%
)
echo Tuned Experiment Completed.
