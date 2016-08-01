for i in `seq 0 19`
do
  for j in 1 3 5
  do
    echo $i
    # optimal policy query
    python rockSampleExp.py -r $i -a OPT-POLICY -t $j
    python rockSampleExp.py -r $i -a OPT-POLICY-ACT -t $j
    # optimal action query
    python rockSampleExp.py -r $i -a JQTP -t $j
    # greedy construction of policy queries
    python rockSampleExp.py -r $i -a MILP-QI-POLICY -t $j
    python rockSampleExp.py -r $i -a MILP-POLICY -t $j
    # action queries by greedy construction of policy queries
    python rockSampleExp.py -r $i -a MILP-QI -t $j
    python rockSampleExp.py -r $i -a MILP -t $j
    # active sampling
    python rockSampleExp.py -r $i -a AS -t $j
    # random query
    python rockSampleExp.py -r $i -a RQ -t $j
  done
done
