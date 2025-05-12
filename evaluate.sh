#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/batch128/AresGW_e15/test.hdf \
AresGW/results_my/month/batch128/AresGWBN_e15/test.hdf \
AresGW/results_my/month/batch128/AresGW350Hz_e15/test.hdf \
AresGW/results_my/month/batch128/MoEk1AresGW_strided_e18/test.hdf \
AresGW/results_my/month/batch128/MoEk2AresGW_strided_e11/test.hdf \
--labels AresGW AresGW_e15 AresGWBN_e15 AresGW350Hz_e15  MoEk1 MoEk2  \
--output AresGW/tests/test_AresGW.png --no-tex --force

'''AresGW/results_my/month/batch128/MoEAresGW_e15/test.hdf \

AresGW/results_my/month/batch128/WavTest_e15/test.hdf \
AresGW/results_my/month/batch128/WavAresGWbase_e15/test.hdf \

MoEAresGW_e15 MoEk2 WavAresGW_e15 WavTest_e15 
'''
