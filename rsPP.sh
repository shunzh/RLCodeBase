for i in `seq 0 9`
do
  echo $i
  # optimal policy query: TOO SLOW!
  for agent in MILP-SIMILAR MILP-SIMILAR-VARIATION MILP-SIMILAR-DISAGREE MILP-SIMILAR-RANDOM
  do
    python rockSampleExp.py -r $i -a $agent
    python rockSampleExp.py -r $i -a $agent -n 2
    python rockSampleExp.py -r $i -a $agent -k 3
    python rockSampleExp.py -r $i -a $agent -t 2
    python rockSampleExp.py -r $i -a $agent -t 4
  done
done

