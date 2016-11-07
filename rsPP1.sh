for i in `seq 0 19`
do
  echo $i

  for agent in MILP-POLICY
  do
    python rockSampleExp.py -r $i -a $agent -y 1 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 1 -k 3 -t 5
    python rockSampleExp.py -r $i -a $agent -y 1 -k 4 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 3 -t 5
    python rockSampleExp.py -r $i -a $agent -y 2 -k 4 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 2 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 3 -t 5
    python rockSampleExp.py -r $i -a $agent -y 3 -k 4 -t 5
  done
done

