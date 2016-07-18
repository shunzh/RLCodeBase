for i in `seq 0 19`
do
for j in 3 7
do
  echo $i
  # optimal policy query
  python rockSampleExp.py -r $i -a OPT-POLICY -n $j
  # optimal action query
  python rockSampleExp.py -r $i -a JQTP -n $j
  # greedy construction of policy queries
  python rockSampleExp.py -r $i -a MILP-QI-POLICY -n $j
  python rockSampleExp.py -r $i -a MILP-POLICY -n $j
  # action queries by greedy construction of policy queries
  python rockSampleExp.py -r $i -a MILP-QI -n $j
  python rockSampleExp.py -r $i -a MILP -n $j
  # active sampling
  python rockSampleExp.py -r $i -a AS -n $j
  # random query
  python rockSampleExp.py -r $i -a RQ -n $j
done
done
