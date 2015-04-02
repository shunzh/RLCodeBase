#!/bin/bash

iterations="5000"

# train each modules using discrete q tables
# -a q: q learning
# -g simple: in `simple` domain
# -c *: category
# -k $iterations: number iterations in q-learning, number defined above
# -e 0.2: exploration rate
# -q: quite running (removing this shows GUI)
python humanWorldExperiment.py -a q -g simple -c targs -k $iterations -e 0.2 -q
python humanWorldExperiment.py -a q -g simple -c segs -k $iterations -e 0.2 -q
python humanWorldExperiment.py -a q -g simple -c obsts -k  $iterations -e 0.2 -q

# move learned q tables to learnedValues for IRL
mv humanAgent*Values.pkl learnedValues
