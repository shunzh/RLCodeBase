for r in `seq 0 99`
do
  for n in 20 25
  do
    for k in 1 2 3 4
    do
      python officeNavigation.py -n $n -k $k -r $r
    done
  done

  for n in 20 25
  do
    for k in 1 2 3 4
    do
      python officeNavigation.py -n $n -k $k -r $r -c
    done
  done
done
