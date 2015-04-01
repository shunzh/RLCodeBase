#!/bin/bash

# train each modules using discrete q tables
# -a q: q learning
# -g simple: in `simple` domain
# -c *: category
# -k 1000: 1000 iterations in q-learning
# -e 0.2: exploration rate
# -q: quite running (removing this shows GUI)
python humanWorldExperiment.py -a q -g simple -c targs -k 1000 -e 0.2 -q
python humanWorldExperiment.py -a q -g simple -c segs -k 1000 -e 0.2 -q
python humanWorldExperiment.py -a q -g simple -c obsts -k 1000 -e 0.2 -q

# move learned q tables to learnedValues for IRL
mv humanAgent*Values.pkl learnedValues
