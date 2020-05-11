for i in `seq 24 39`
do
  echo $i
  n=8

  for agent in OPT-POLICY MILP-POLICY MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -n $n -r $i -a $agent -y 1 -k 2
    echo "done"
    python rockSampleExp.py -n $n -r $i -a $agent -y 2 -k 2
    echo "done"
    python rockSampleExp.py -n $n -r $i -a $agent -y 3 -k 2
    echo "done"
    python rockSampleExp.py -n $n -r $i -a $agent -y 1 -k 3
    echo "done"
    python rockSampleExp.py -n $n -r $i -a $agent -y 2 -k 3
    echo "done"
    python rockSampleExp.py -n $n -r $i -a $agent -y 3 -k 3
    echo "done"
  done
done

