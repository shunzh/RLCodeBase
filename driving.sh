for i in `seq 0 4`
do
  echo $i

  #for agent in MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE
  for agent in MILP-POLICY RAND-POLICY
  do
    python policyGradExp.py -a $agent -r $i &
  done
done

