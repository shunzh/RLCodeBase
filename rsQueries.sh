#!/bin/bash
flag="-P queries"

python rockSampleExp.py -a AQTP-P -r $1 -l 0 $flag
echo "-1"
python rockSampleExp.py -a AQTP-P -r $1 -l 5 $flag
echo "-1"
python rockSampleExp.py -a AQTP-P -r $1 -l 10 $flag
echo "-1"
python rockSampleExp.py -a AQTP-P -r $1 -l 15 $flag
echo "-1"
python rockSampleExp.py -a AQTP-P -r $1 -l 20 $flag
echo "-1"
