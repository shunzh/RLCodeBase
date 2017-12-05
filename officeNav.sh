for r in `seq 0 299`
do
  for n in 15
  do
    for k in 1 2 3 4
    do
      python officeNavigation.py -n $n -k $k -r $r -c
    done
  done
done
