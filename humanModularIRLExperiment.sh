#!/bin/bash

python inverseModularRLExperiments.py
mv values.pkl learnedValues

# clear previous recorded stats
rm stats
# draw figures
for i in `seq 0 31`;
do
  python humanWorldExperiment.py -a ModularDiscrete -g vr$i
  #python humanWorldExperiment.py -a Modular -g vr$i
done

# convert from eps to png
for f in `ls task*.eps`; do
  convert $f -density 100 -flatten ${f%.*}.png;
done

# plot figures on number of contected objects
# this uses the `stat` file
python plotContacts.py
# print the learned value again
python printResults.py

rm *.eps
