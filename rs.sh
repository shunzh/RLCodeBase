for i in `seq 0 19`
do
  # optimal policy query
  python rockSampleExp.py -r $i -a OPT-POLICY
  # optimal action query
  python rockSampleExp.py -r $i -a JQTP
  # greedy construction of policy queries
  python rockSampleExp.py -r $i -a MILP-QI-POLICY
  python rockSampleExp.py -r $i -a MILP-POLICY
  # action queries by greedy construction of policy queries
  python rockSampleExp.py -r $i -a MILP-QI
  python rockSampleExp.py -r $i -a MILP
  # active sampling
  python rockSampleExp.py -r $i -a AS
  # random query
  python rockSampleExp.py -r $i -a RQ
done
