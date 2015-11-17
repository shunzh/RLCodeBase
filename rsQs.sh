#!/bin/bash
flag="-p qs"

for i in `seq 0 19`;
do
  python rockSampleExp.py -a JQTP -r $i $flag
done
