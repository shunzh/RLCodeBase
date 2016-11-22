for i in `seq 0 19`
do
  echo $i

  for agent in OPT-POLICY MILP-POLICY
  #for agent in MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -r $i -a $agent -y 1 -k 2
    python rockSampleExp.py -r $i -a $agent -y 2 -k 2
    python rockSampleExp.py -r $i -a $agent -y 3 -k 2
    python rockSampleExp.py -r $i -a $agent -y 1 -k 3
    python rockSampleExp.py -r $i -a $agent -y 2 -k 3
    python rockSampleExp.py -r $i -a $agent -y 3 -k 3
    python rockSampleExp.py -r $i -a $agent -y 1 -k 4
    python rockSampleExp.py -r $i -a $agent -y 2 -k 4
    python rockSampleExp.py -r $i -a $agent -y 3 -k 4
  done
done

