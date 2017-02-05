for i in `seq 10 19`
do
  for j in 10 50
  do
    echo $i
    #python rockSampleExp.py -r $i -a FEAT-GREEDY -n $j
    #python rockSampleExp.py -r $i -a FEAT-RANDOM -n $j
    python rockSampleExp.py -r $i -a MILP-POLICY -n $j &
  done
done
