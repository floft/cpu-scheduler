#!/bin/bash
#
# Make sure debugTiming = debug = True in simulation{,_old}.py
#
rm results.{before,after}/0_debug_fcfs_cpu3.csv
time python simulation.py > output1.txt
time python simulation_old.py > output2.txt
sed '/Executing /d' output1.txt > output1_small.txt
sed '/Executing /d' output2.txt > output2_small.txt
sed '/Incrementing /d' output1_small.txt > output1_small2.txt
vim output1_small2.txt output2_small.txt
