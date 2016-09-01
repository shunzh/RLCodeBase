for i in `seq 0 19`
do
  echo $i
  # optimal policy query: TOO SLOW!
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY-ACT
  # optimal action query
  python rockSampleKWayExp.py -r $i -a JQTP
  # greedy construction of policy queries
  python rockSampleKWayExp.py -r $i -a MILP-QI-POLICY
  python rockSampleKWayExp.py -r $i -a MILP-POLICY
  # action queries by greedy construction of policy queries
  python rockSampleKWayExp.py -r $i -a MILP-QI
  python rockSampleKWayExp.py -r $i -a MILP
  # active sampling
  python rockSampleKWayExp.py -r $i -a AS
  # random query
  python rockSampleKWayExp.py -r $i -a RQ
done
