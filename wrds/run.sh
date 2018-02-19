#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M yzhanggm@connect.ust.hk

echo "Starting job at `date`"

RESULT_FILE=result_info.txt

#if [ -fe $RESULT_FILE ]
#then
#    rm $RESULT_FILE
#fi

~/miniconda3/bin/python3 grid_research.py >> ./$RESULT_FILE

echo "Finished job at `date`"