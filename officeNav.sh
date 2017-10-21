for n in 10
do
  for k in 0 1 2
  do
    python officeNavigation.py -n $n -k $k -a brute
    python officeNavigation.py -n $n -k $k -a alg1
    python officeNavigation.py -n $n -k $k -a chain
    python officeNavigation.py -n $n -k $k -a random
    python officeNavigation.py -n $n -k $k -a nq
  done
done
