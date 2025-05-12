#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/batch128/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_ds_e18/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_expand_e20/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_less_e19/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_more_e13/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided50e_e45/test.hdf \
--labels AresGW AresGW_e15 MoEk2ds_18 MoEk2expand_20  MoEk2less_19 MoEk2more_13 MoEk1_45 \
--output AresGW/tests/test_AresGW.png --no-tex --xlim 1 1000 --ylim 1100 2000 --force




