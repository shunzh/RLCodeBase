for i in `seq 0 999`
do
  for p in 0.1 0.3 0.5 0.7 0.9
  do
    python officeNavigation.py -p $p -r $i
  done
done

