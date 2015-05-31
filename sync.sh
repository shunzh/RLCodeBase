#!/bin/bash

./clean.sh
# git push/pull is fussy. just sync it 
rsync -a . $ut:~/workspace/Modular/
