#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/batch128/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided50e_e45/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_ds_e12/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_ds_e18/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_more_e13/test.hdf \
AresGW/results_my/month/batch128/MoE8ek2AresGW_strided_e16/test.hdf \
AresGW/results_my/month/batch128/MoEk1BNAresGW_strided_e28/test.hdf \
--labels AresGW AresGW_e15 MoEk1_E45 MoEk2ds_12 MoEk2ds_18 MoEk2more_13 MoE8ek2_16 MoEk1BN_E28 \
--output AresGW/tests/test_AresGW.png --no-tex --xlim 1 1000 --ylim 1100 2000 --force




