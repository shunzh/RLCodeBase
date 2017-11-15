for r in `seq 0 99`
do
  for n in 15
  do
    for k in 0 1 3
    do
      #python officeNavigation.py -n $n -k $k -r $r -a brute
      python officeNavigation.py -n $n -k $k -r $r -a alg1
      python officeNavigation.py -n $n -k $k -r $r -a chain
      python officeNavigation.py -n $n -k $k -r $r -a random
      python officeNavigation.py -n $n -k $k -r $r -a relevantRandom
      python officeNavigation.py -n $n -k $k -r $r -a nq
    done
  done

  for n in 15
  do
    for k in 0 1 3
    do
      #python officeNavigation.py -n $n -k $k -r $r -c -a brute
      python officeNavigation.py -n $n -k $k -r $r -c -a alg1
      python officeNavigation.py -n $n -k $k -r $r -c -a chain
      python officeNavigation.py -n $n -k $k -r $r -c -a random
      python officeNavigation.py -n $n -k $k -r $r -c -a relevantRandom
      python officeNavigation.py -n $n -k $k -r $r -c -a nq
    done
  done
done
