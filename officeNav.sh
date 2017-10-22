for n in 5
do
  for k in 0 1 2 3
  do
    for r in `seq 0 9`
    do
      python officeNavigation.py -n $n -k $k -r $r -a brute
      python officeNavigation.py -n $n -k $k -r $r -a alg1
      python officeNavigation.py -n $n -k $k -r $r -a chain
      python officeNavigation.py -n $n -k $k -r $r -a random
      python officeNavigation.py -n $n -k $k -r $r -a nq
    done
  done
done
