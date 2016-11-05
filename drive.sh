for i in `seq 0 19`
do
  echo $i
  for agent in MILP-SIMILAR MILP-SIMILAR-VARIATION MILP-SIMILAR-DISAGREE MILP-SIMILAR-RANDOM
  do
    python drivingExp.py -r $i -a $agent 
    python drivingExp.py -r $i -a $agent -k 3
    python drivingExp.py -r $i -a $agent -n 2
    python drivingExp.py -r $i -a $agent -n 2 -k 3
  done
done

