#!/bin/bash
for i in `seq 0 31`;
do
  python humanWorld.py -a Modular -g vr$i
done  
