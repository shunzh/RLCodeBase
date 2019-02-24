for i in `seq 0 499`
do
  for p in `seq 0 9`
  do
    python officeNavigation.py -p 0.$p -r $i
  done
  python officeNavigation.py -p 1.0 -r $i
done

