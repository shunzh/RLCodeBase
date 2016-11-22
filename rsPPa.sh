for i in `seq 0 19`
do
  echo $i

  for agent in APRIL3 #APRIL1 APRIL2
  do
    python rockSampleExp.py -r $i -a $agent -n 10 -y 1 -k 2
    python rockSampleExp.py -r $i -a $agent -n 10 -y 2 -k 2
    python rockSampleExp.py -r $i -a $agent -n 10 -y 3 -k 2
  done
done

