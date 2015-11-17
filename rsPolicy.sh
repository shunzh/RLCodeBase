#!/bin/bash
flag="-p"

python rockSampleExp.py -a JQTP -r $1 -l 5 $flag
python rockSampleExp.py -a JQTP -r $1 -l 10 $flag
python rockSampleExp.py -a JQTP -r $1 -l 15 $flag

python rockSampleExp.py -a AQTP -r $1 -l 5 $flag
python rockSampleExp.py -a AQTP -r $1 -l 10 $flag
python rockSampleExp.py -a AQTP -r $1 -l 15 $flag
