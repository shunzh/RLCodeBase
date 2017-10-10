for n in 10
do
  for p in 0.1 0.5 0.9
  do
    for k in 2
    do
      python officeNavigation.py -n $n -k $k -p $p -a alg1
      python officeNavigation.py -n $n -k $k -p $p -a chain
      python officeNavigation.py -n $n -k $k -p $p -a random
      python officeNavigation.py -n $n -k $k -p $p -a nq
    done
  done
done
