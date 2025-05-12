#!/bin/bash
cd ..

python3 MLGWSC/contributions/sensitivity_plot.py \
--files AresGW/results_best/test_eval_output_s24w6d1_1.hdf \
AresGW/results_my/month/paper/MoEAresGW_e14/test.hdf \
AresGW/results_my/month/paper/MoEAresGW_e18/test.hdf \
--labels AresGW_batch400 MoEAresGW_e14 MoEAresGW_e18 \
--output AresGW/tests/test_AresGW.png --no-tex --xlim 1 1000 --ylim 1100 2000 --force




