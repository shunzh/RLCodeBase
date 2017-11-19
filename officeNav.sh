for r in `seq 11 99`
do
  for n in 20
  do
    for k in 0 1 2 3
    do
      python officeNavigation.py -n $n -k $k -r $r
    done
  done

  for n in 20
  do
    for k in 0 1 2 3
    do
      python officeNavigation.py -n $n -k $k -r $r -c
    done
  done
done
