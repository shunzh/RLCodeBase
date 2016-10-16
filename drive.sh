for i in `seq 20 49`
do
  echo $i
  #python drivingExp.py -r $i -a OPT-POLICY
  python drivingExp.py -r $i -a MILP-SIMILAR
  python drivingExp.py -r $i -a MILP-SIMILAR-NAIVE
done

