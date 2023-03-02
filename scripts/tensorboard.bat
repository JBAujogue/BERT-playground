@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0..
call conda activate transformers_nlp
tensorboard --logdir=logs
call conda deactivate
pause