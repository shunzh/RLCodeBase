for i in `seq 0 19`
do
  echo $i
  #python drivingExp.py -r $i -a MILP-SIMILAR -t 1
  #python drivingExp.py -r $i -a MILP-SIMILAR-NAIVE -t 1
  python drivingExp.py -r $i -a JQTP
  python drivingExp.py -r $i -a MILP
  python drivingExp.py -r $i -a AS
done

