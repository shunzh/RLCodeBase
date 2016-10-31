for i in `seq 0 19`
do
  echo $i
  #python drivingExp.py -r $i -a OPT-POLICY
  python drivingExp.py -r $i -a MILP-SIMILAR
  python drivingExp.py -r $i -a MILP-SIMILAR-VARIATION
  python drivingExp.py -r $i -a MILP-SIMILAR-DISAGREE
  python drivingExp.py -r $i -a MILP-SIMILAR-RANDOM
done

