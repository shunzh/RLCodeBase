for n in 10
do
  #for p in 0.1 0.5 0.9
  for p in 0.2 0.4 0.6 0.8
  do
    for k in 1
    do
      for r in `seq 0 19`
      do
        python officeNavigation.py -n $n -k $k -p $p -r $r -a alg1
        python officeNavigation.py -n $n -k $k -p $p -r $r -a chain
        python officeNavigation.py -n $n -k $k -p $p -r $r -a random
        python officeNavigation.py -n $n -k $k -p $p -r $r -a nq
      done
    done
  done
done
