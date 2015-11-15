#!/bin/bash
flag="-p queries"

python rockSampleExp.py -a AQTP -r $1 -l 0 $flag
echo "-1"
python rockSampleExp.py -a AQTP -r $1 -l 10 $flag
echo "-1"
python rockSampleExp.py -a AQTP -r $1 -l 20 $flag
echo "-1"
