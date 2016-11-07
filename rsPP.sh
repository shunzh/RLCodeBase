for i in `seq 0 19`
do
  echo $i

  for agent in MILP-POLICY MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -r $i -a $agent -y 1 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 1 -k 5 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 5 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 5 -t 5
    python rockSampleExp.py -r $i -a $agent -y 1 -k 8 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 8 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 8 -t 5
  done
done

