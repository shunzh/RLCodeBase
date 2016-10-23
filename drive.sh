for i in `seq 0 19`
do
  echo $i
  python drivingExp.py -r $i -a OPT-POLICY
  python drivingExp.py -r $i -a MILP-SIMILAR -t 2
  python drivingExp.py -r $i -a MILP-SIMILAR -t 3
  python drivingExp.py -r $i -a MILP-SIMILAR -t 4
  python drivingExp.py -r $i -a MILP-SIMILAR -t 5
  python drivingExp.py -r $i -a MILP-SIMILAR-VARIATION -t 2
  python drivingExp.py -r $i -a MILP-SIMILAR-VARIATION -t 3
  python drivingExp.py -r $i -a MILP-SIMILAR-VARIATION -t 4
  python drivingExp.py -r $i -a MILP-SIMILAR-VARIATION -t 5
  python drivingExp.py -r $i -a MILP-SIMILAR-RANDOM -t 2
  python drivingExp.py -r $i -a MILP-SIMILAR-RANDOM -t 3
  python drivingExp.py -r $i -a MILP-SIMILAR-RANDOM -t 4
  python drivingExp.py -r $i -a MILP-SIMILAR-RANDOM -t 5
done

