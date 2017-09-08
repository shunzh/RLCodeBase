for i in `seq 0 29`
do
  echo $i
  #python officeNavigation.py -r $i -a brute
  python officeNavigation.py -r $i
done
