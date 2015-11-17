#!/bin/bash
flag="-q test"

python rockSampleExp.py -a JQTP -r $1 -l 0 $flag
python rockSampleExp.py -a JQTP -r $1 -l 5 $flag
python rockSampleExp.py -a JQTP -r $1 -l 10 $flag
python rockSampleExp.py -a JQTP -r $1 -l 15 $flag
python rockSampleExp.py -a JQTP -r $1 -l 20 $flag

python rockSampleExp.py -a AQTP -r $1 -l 0 $flag
python rockSampleExp.py -a AQTP -r $1 -l 5 $flag
python rockSampleExp.py -a AQTP -r $1 -l 10 $flag
python rockSampleExp.py -a AQTP -r $1 -l 15 $flag
python rockSampleExp.py -a AQTP -r $1 -l 20 $flag

python rockSampleExp.py -a AQTP-NF -r $1 -l 0 $flag
python rockSampleExp.py -a AQTP-NF -r $1 -l 5 $flag
python rockSampleExp.py -a AQTP-NF -r $1 -l 10 $flag
python rockSampleExp.py -a AQTP-NF -r $1 -l 15 $flag
python rockSampleExp.py -a AQTP-NF -r $1 -l 20 $flag

# python rockSampleExp.py -a PTP -r $1 -l 0 $flag
# python rockSampleExp.py -a PTP -r $1 -l 5 $flag
# python rockSampleExp.py -a PTP -r $1 -l 10 $flag
# python rockSampleExp.py -a PTP -r $1 -l 15 $flag
# python rockSampleExp.py -a PTP -r $1 -l 20 $flag
# 
# python rockSampleExp.py -a RQ -r $1 -l 0 $flag
# python rockSampleExp.py -a RQ -r $1 -l 5 $flag
# python rockSampleExp.py -a RQ -r $1 -l 10 $flag
# python rockSampleExp.py -a RQ -r $1 -l 15 $flag
# python rockSampleExp.py -a RQ -r $1 -l 20 $flag
# 
# python rockSampleExp.py -a NQ -r $1
