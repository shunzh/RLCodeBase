for i in `seq 0 19`
do
  echo $i
  #python rockSampleExp.py -r $i -a OPT-POLICY
  #python rockSampleExp.py -r $i -a OPT-POLICY -k 3

  for agent in MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -r $i -a $agent
    python rockSampleExp.py -r $i -a $agent -y 1
    python rockSampleExp.py -r $i -a $agent -y 2
    python rockSampleExp.py -r $i -a $agent -k 3
    python rockSampleExp.py -r $i -a $agent -k 3 -y 1
    python rockSampleExp.py -r $i -a $agent -k 3 -y 2
  done
done

