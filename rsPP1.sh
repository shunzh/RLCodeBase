for i in `seq 0 19`
do
  echo $i

  t=2
  for agent in MILP-POLICY MILP-SIMILAR SIMILAR-VARIATION SIMILAR-DISAGREE SIMILAR-RANDOM
  do
    python rockSampleExp.py -n 10 -r $i -a $agent -y 1 -k 2 -t $t
    python rockSampleExp.py -n 10 -r $i -a $agent -y 2 -k 2 -t $t
    python rockSampleExp.py -n 10 -r $i -a $agent -y 3 -k 2 -t $t
    #python rockSampleExp.py -n 10 -r $i -a $agent -y 1 -k 5
    #python rockSampleExp.py -n 10 -r $i -a $agent -y 2 -k 5
    #python rockSampleExp.py -n 10 -r $i -a $agent -y 3 -k 5
    python rockSampleExp.py -n 10 -r $i -a $agent -y 1 -k 8 -t $t
    python rockSampleExp.py -n 10 -r $i -a $agent -y 2 -k 8 -t $t
    python rockSampleExp.py -n 10 -r $i -a $agent -y 3 -k 8 -t $t
  done
done

