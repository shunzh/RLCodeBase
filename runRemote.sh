#!/bin/bash

# git push/pull is fussy. just sync it 
rsync -a . $ut:~/workspace/Modular/

# run command at remote
ssh -X $ut 'cd workspace/Modular; python inverseModularRLExperiments.py'

# get back the results
rsync -a $ut:~/workspace/Modular/values.pkl .
