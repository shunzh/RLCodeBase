for r in `seq 0 299`
do
  for n in 10
  do
    for k in 0 1 2 3 4 5 6 7 8 9
    do
      python officeNavigation.py -n $n -k $k -r $r -c
    done
  done
done
