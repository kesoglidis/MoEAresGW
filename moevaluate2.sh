#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/batch128/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided_e18/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided_e22/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided_e25/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided40e_e40/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided50e_e41/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided50e_e45/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_strided_e11/test.hdf \
AresGW/results_my/month/batch128/MoEk1BNAresGW_strided_e28/test.hdf \
--labels AresGW AresGW_e15 MoEk1_E18 MoEk1_E22 MoEk1_E25 MoEk1_E40 MoEk1_E41 MoEk1_E45 MoEk2_11 MoEk1BN_E28 \
--output AresGW/tests/test_AresGW.png --no-tex --force




