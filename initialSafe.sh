for i in `seq 0 499`
do
  for p in 0.05 0.3 0.5 0.7 0.95
  do
    python officeNavigation.py -p $p -r $i
  done
done

