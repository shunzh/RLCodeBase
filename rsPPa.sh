for i in `seq 0 19`
do
  echo $i

  for agent in APRIL0
  do
    python rockSampleExp.py -r $i -a $agent -y 1 -k 2
    python rockSampleExp.py -r $i -a $agent -y 2 -k 2
    python rockSampleExp.py -r $i -a $agent -y 3 -k 2
  done
done

