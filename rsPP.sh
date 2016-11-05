for i in `seq 0 19`
do
  echo $i
  #python rockSampleExp.py -r $i -a OPT-POLICY
  #python rockSampleExp.py -r $i -a OPT-POLICY -k 3

  for agent in MILP-POLICY MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -r $i -a $agent -y 1
    python rockSampleExp.py -r $i -a $agent -y 2
    python rockSampleExp.py -r $i -a $agent -y 3
  done
done

