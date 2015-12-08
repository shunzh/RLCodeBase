#!/bin/bash
flag=""

python nWayCorridorExp.py -a JQTP -r $1 -s 6 -l 3 $flag
python nWayCorridorExp.py -a JQTP -r $1 -s 10 -l 5 $flag
python nWayCorridorExp.py -a JQTP -r $1 -s 20 -l 10 $flag

python nWayCorridorExp.py -a AQTP -r $1 -s 6 -l 3 $flag
python nWayCorridorExp.py -a AQTP -r $1 -s 10 -l 5 $flag
python nWayCorridorExp.py -a AQTP -r $1 -s 20 -l 10 $flag

python nWayCorridorExp.py -a AQTP-NF -r $1 -s 6 -l 3 $flag
python nWayCorridorExp.py -a AQTP-NF -r $1 -s 10 -l 5 $flag
python nWayCorridorExp.py -a AQTP-NF -r $1 -s 20 -l 10 $flag
