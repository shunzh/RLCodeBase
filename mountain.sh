for i in `seq 10 12`
do
  echo $i

  for agent in MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE
  do
    python mountainCarExp.py -r $i -a $agent
  done
done

