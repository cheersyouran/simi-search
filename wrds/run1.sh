#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M yzhanggm@connect.ust.hk

echo "Starting job at `date`"

RESULT_FILE=result_info_run1.txt

#if [ -fe $RESULT_FILE ]
#then
#    rm $RESULT_FILE
#fi

~/miniconda3/bin/python3 ../codes/regression_test1.py >> $RESULT_FILE

echo "Finished job at `date`"