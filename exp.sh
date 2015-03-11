#!/bin/bash
mv values.pkl learnedValues

rm stats
# draw figures
for i in `seq 0 31`;
do
  python humanWorld.py -a ModularV -g vr$i
done

# convert to png
for f in `ls task*.eps`; do
  convert $f -density 100 -flatten ${f%.*}.png;
done

python plotContacts.py
python printResults.py

rm *.eps
