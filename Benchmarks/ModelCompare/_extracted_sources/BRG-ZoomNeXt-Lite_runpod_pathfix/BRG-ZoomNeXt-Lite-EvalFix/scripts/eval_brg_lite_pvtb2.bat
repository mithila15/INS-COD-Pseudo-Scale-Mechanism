@echo off
cd /d %~dp0\..
python main_for_brg_lite.py ^
  --config configs/brg_lite_pvtb2_train.py ^
  --data-cfg configs/dataset_cod_windows.yaml ^
  --model-name PvtV2B2_BRG_ZoomNeXt_Lite ^
  --load-from outputs_brg_lite\YOUR_EXP\exp_0\pth\state_final.pth ^
  --evaluate ^
  --save-results
