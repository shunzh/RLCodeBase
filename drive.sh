for i in `seq 0 19`
do
  echo $i
  # optimal policy query: TOO SLOW!
  #python rockSampleKWayExp.py -r $i -a OPT-POLICY
  python drivingExp.py -r $i -a MILP-SIMILAR
  python drivingExp.py -r $i -a MILP-SIMILAR-NAIVE
done

