#!/bin/bash

mkdir results_my/$3/$1_e$2
python3 test.py dataset-4/v2/$3/test_foreground_s24w6d1_1.hdf results_my/$3/$1_e$2/test_fgevents_s24w6d1_1.hdf --weights /home/tzagkari/Documents/thesis/AresGW/runs/$1/checkpoints/net_epoch_$2.pt


python3 test.py dataset-4/v2/$3/test_background_s24w6d1_1.hdf results_my/$3/$1_e$2/test_bgevents_s24w6d1_1.hdf --weights /home/tzagkari/Documents/thesis/AresGW/runs/$1/checkpoints/net_epoch_$2.pt

cd ..

MLGWSC/evaluate.py --injection-file AresGW/dataset-4/v2/$3/test_injections_s24w6d1_1.hdf --foreground-events AresGW/results_my/$3/$1_e$2/test_fgevents_s24w6d1_1.hdf \
--foreground-files AresGW/dataset-4/v2/$3/test_foreground_s24w6d1_1.hdf \
--background-events AresGW/results_my/$3/$1_e$2/test_bgevents_s24w6d1_1.hdf \
--output-file AresGW/results_my/$3/$1_e$2/test.hdf --force --verbose


