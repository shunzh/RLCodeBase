#!/bin/bash

iterations="5000"
exploration='0.5'
eligibilityTraces='0'
discounter='0.9'
learningRate='0.1'

# train each modules using discrete q tables
# -a q: q learning
# -g simple: in `simple` domain
# -c *: category
# -k $iterations: number iterations in q-learning, number defined above
# -e 0.2: exploration rate
# -q: quite running (removing this shows GUI)
python humanWorldExperiment.py -a q -g simple -c targs -k $iterations -e $exploration -t $eligibilityTraces -d $discounter -l $learningRate -q
python humanWorldExperiment.py -a q -g simple -c segs -k $iterations -e $exploration -t $eligibilityTraces -d $discounter -l $learningRate -q
python humanWorldExperiment.py -a q -g simple -c obsts -k 1000 -e $exploration -t $eligibilityTraces -d $discounter -l $learningRate -q

# move learned q tables to learnedValues for IRL
mv humanAgent*Values.pkl learnedValues
