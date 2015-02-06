#!/bin/bash
mv weights.pkl learnedValues

# draw figures
for i in `seq 0 31`;
do
  python humanWorld.py -a Modular -g vr$i
done

# convert to png
for f in `ls task*.eps`; do
  convert $f -density 100 -flatten ${f%.*}.png;
done

python printResults.py
