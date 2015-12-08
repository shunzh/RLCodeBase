#!/bin/bash
flag=""

python nWayCorridorExp.py -a JQTP -r $1 -s 8 $flag
python nWayCorridorExp.py -a JQTP -r $1 -s 4 $flag
python nWayCorridorExp.py -a JQTP -r $1 -s 2 $flag

python nWayCorridorExp.py -a AQTP-NF -r $1 -s 8 $flag
python nWayCorridorExp.py -a AQTP-NF -r $1 -s 4 $flag
python nWayCorridorExp.py -a AQTP-NF -r $1 -s 2 $flag

