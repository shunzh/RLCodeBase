#!/bin/bash
flag="-c"

python rockSampleExp.py -a JQTP -r $1 -l 0 $flag
python rockSampleExp.py -a JQTP -r $1 -l 10 $flag
python rockSampleExp.py -a JQTP -r $1 -l 20 $flag

python rockSampleExp.py -a AQTP -r $1 -l 0 $flag
python rockSampleExp.py -a AQTP -r $1 -l 10 $flag
python rockSampleExp.py -a AQTP -r $1 -l 20 $flag

python rockSampleExp.py -a TPNQ -r $1 -l 0 $flag
python rockSampleExp.py -a TPNQ -r $1 -l 10 $flag
python rockSampleExp.py -a TPNQ -r $1 -l 20 $flag

python rockSampleExp.py -a RQ -r $1 -l 0 $flag
python rockSampleExp.py -a RQ -r $1 -l 10 $flag
python rockSampleExp.py -a RQ -r $1 -l 20 $flag

python rockSampleExp.py -a NQ -r $1

