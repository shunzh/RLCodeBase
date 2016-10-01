for i in `seq 0 19`
do
  echo $i
  # optimal policy query: TOO SLOW!
  python rockSampleKWayExp.py -r $i -a OPT-POLICY
  python rockSampleKWayExp.py -r $i -a OPT-TRAJ-SIMILAR
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY-ACT -t $j
  # optimal action query
  python rockSampleKWayExp.py -r $i -a JQTP
  # greedy construction of policy queries
  #python rockSampleKWayExp.py -r $i -a MILP-QI-POLICY -t $j
  #python rockSampleKWayExp.py -r $i -a MILP-POLICY -t $j
  # action queries by greedy construction of policy queries
  #python rockSampleKWayExp.py -r $i -a MILP-QI -t $j
  #python rockSampleKWayExp.py -r $i -a MILP -t $j
  # active sampling
  #python rockSampleKWayExp.py -r $i -a AS -t $j
  # random query
  #python rockSampleKWayExp.py -r $i -a RQ -t $j
done
