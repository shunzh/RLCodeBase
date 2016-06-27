for i in `seq 0 9`
do
  python rockSampleExp.py -r $i -a AS -m 3
  python rockSampleExp.py -r $i -a AS -m 5
  python rockSampleExp.py -r $i -a MILP -m 3
  python rockSampleExp.py -r $i -a MILP -m 5
done
