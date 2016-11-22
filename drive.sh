for i in `seq 0 19`
do
  echo $i
  for agent in MILP-POLICY MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python drivingExp.py -r $i -a $agent 
    python drivingExp.py -r $i -a $agent -k 3
    python drivingExp.py -r $i -a $agent -k 4
  done
done

