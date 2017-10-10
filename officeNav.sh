for n in 5 10 15
do
  for k in 2 3 4
  do
    if [ "$n" != "15" ]; then
      python officeNavigation.py -n $n -k $k -a brute
    fi
    python officeNavigation.py -n $n -k $k -a alg1
    python officeNavigation.py -n $n -k $k -a chain
    python officeNavigation.py -n $n -k $k -a random
    python officeNavigation.py -n $n -k $k -a nq
  done
done
