#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/batch400/AresGW_e14/test.hdf \
AresGW/results_my/month/batch400/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_ds_e18/test.hdf \
AresGW/results_my/month/batch400/MoEk2AresGW_ds_e18/test.hdf \
--labels AresGW_batch400 myAresGW_batch400_e14 myAresGW_batch400_e15 AresGW_batch128 MoEk2ds_batch128_18 MoEk2ds_batch400_e18 \
--output AresGW/tests/test_AresGW.png --no-tex --xlim 1 1000 --ylim 1100 2000 --force




