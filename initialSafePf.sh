for i in `seq 0 499`
do
  for n in 8 9 10 11 12
  do
    python officeNavigation.py -n $n -r $i
  done
done

