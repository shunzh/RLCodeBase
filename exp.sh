#!/bin/bash
for i in `seq 0 30`;
do
  python humanWorld.py -a Modular -g vr$i
done  
