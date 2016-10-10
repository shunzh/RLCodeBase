for i in `seq 0 19`
do
  echo $i
  # optimal policy query: TOO SLOW!
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY
  #python rockSampleKWayExp.py -r $i -a OPT-TRAJ-SIMILAR
  #python rockSampleKWayExp.py -r $i -a OPT-TRAJ-SIMILAR -n 1
  #python rockSampleKWayExp.py -r $i -a OPT-TRAJ-SIMILAR -n 0
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY-ACT -t $j
  # optimal action query
  #python rockSampleKWayExp.py -r $i -a JQTP
  # greedy construction of policy queries
  #python rockSampleKWayExp.py -r $i -a MILP-QI-POLICY -t $j
  #python rockSampleKWayExp.py -r $i -a MILP-POLICY
  # action queries by greedy construction of policy queries
  #python rockSampleKWayExp.py -r $i -a MILP-QI -t $j
  #python rockSampleKWayExp.py -r $i -a MILP -t $j
  # active sampling
  #python rockSampleKWayExp.py -r $i -a AS -t $j
  # random query
  #python rockSampleKWayExp.py -r $i -a RQ -t $j

  #python rockSampleKWayExp.py -r $i -a MILP-DEMO
  #python rockSampleKWayExp.py -r $i -a MILP-DEMO-BATCH
  python rockSampleKWayExp.py -r $i -a MILP-SIMILAR
  python rockSampleKWayExp.py -r $i -a MILP-SIMILAR-NAIVE
done

