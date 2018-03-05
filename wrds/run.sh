#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M yzhanggm@connect.ust.hk

echo "Starting job at `date`"

if [ $# -eq 0 ]
  then
    echo "Error! No arguments supplied!"
    exit 1
fi

RESULT_FILE=$1

#if [ -fe $RESULT_FILE ]
#then
#    rm $RESULT_FILE
#fi

~/miniconda3/bin/python3 ../codes/regression_test.py >> $RESULT_FILE

echo "Finished job at `date`"