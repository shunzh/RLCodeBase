for i in `seq 0 19`
do
  echo $i
  python drivingExp.py -r $i -a MILP-SIMILAR -t 3
done

