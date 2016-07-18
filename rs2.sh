for i in `seq 0 19`
do
  for k in 3 4
  do
    echo $i
    # optimal policy query
    python rockSampleExp.py -r $i -a OPT-POLICY -m $k
    # greedy construction of policy queries
    python rockSampleExp.py -r $i -a MILP-QI-POLICY -m $k
    python rockSampleExp.py -r $i -a MILP-POLICY -m $k
  done
done
