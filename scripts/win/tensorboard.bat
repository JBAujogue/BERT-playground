@ECHO OFF
call %USERPROFILE%/miniconda3/Scripts/activate.bat
cd %~dp0..
call conda activate bert-playground
tensorboard --logdir=logs
call conda deactivate
pause