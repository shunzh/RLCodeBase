for i in `seq 300 999`
do
  for p in 0 0.3 0.5 0.7 1
  do
    python officeNavigation.py -p $p -r $i
  done
done

