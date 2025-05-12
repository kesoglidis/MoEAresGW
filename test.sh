#!/bin/bash
mkdir results_my/$1
python3 test.py dataset-4/v2/test_foreground_s24w6d1_1.hdf results_my/$1/test_fgevents_s24w6d1_1.hdf --weights /home/tzagkari/Documents/thesis/AresGW/runs/$1/weights.pt

python3 test.py dataset-4/v2/test_background_s24w6d1_1.hdf results_my/$1/test_bgevents_s24w6d1_1.hdf --weights /home/tzagkari/Documents/thesis/AresGW/runs/$1/weights.pt

cd ..

MLGWSC/evaluate.py --injection-file AresGW/dataset-4/v2/test_injections_s24w6d1_1.hdf --foreground-events AresGW/results_my/$1/test_fgevents_s24w6d1_1.hdf \
--foreground-files AresGW/dataset-4/v2/test_foreground_s24w6d1_1.hdf \
--background-events AresGW/results_my/$1/test_bgevents_s24w6d1_1.hdf \
--output-file AresGW/results_my/$1/test.hdf --force --verbose


