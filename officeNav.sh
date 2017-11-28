for r in `seq 0 299`
do
  for n in 10 15
  do
    for k in 1 2 3
    do
      python officeNavigation.py -n $n -k $k -r $r
      python officeNavigation.py -n $n -k $k -r $r -c
    done
  done
done
