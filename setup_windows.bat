@echo off
REM Setup script for MoEAresGW on Windows with AMD GPU (ROCm)
REM Requires Python 3.12

set PYTHON=C:\Users\Tzagkari\AppData\Local\Programs\Python\Python312\python.exe

echo === Creating virtual environment with Python 3.12 ===
"%PYTHON%" -m venv venv

echo === Activating venv ===
call venv\Scripts\activate.bat

echo === Installing ROCm SDK ===
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

echo === Installing PyTorch with ROCm ===
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl

echo === Installing project requirements ===
pip install -r requirements.txt

echo.
echo === Setup complete! ===
echo.
echo To activate the venv, run:  venv\Scripts\activate.bat
echo To train with GPU, use:     python train.py --train-device cuda ...
echo.
pause
