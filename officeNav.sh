for i in `seq 0 49`
do
  echo $i
  python officeNavigation.py -r $i -a brute
  python officeNavigation.py -r $i -a alg1
  #python officeNavigation.py -r $i -a alg3
done
