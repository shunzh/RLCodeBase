for i in `seq 0 19`
do
  echo $i
  # optimal policy query: TOO SLOW!
  python rockSampleExp.py -r $i -a MILP-SIMILAR
  python rockSampleExp.py -r $i -a MILP-SIMILAR-NAIVE
done

