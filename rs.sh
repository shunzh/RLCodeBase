for i in `seq 0 9`
do
  python rockSampleExp.py -r $i
  python rockSampleExp.py -r $i -a RQ
  python rockSampleExp.py -r $i -a AS
  python rockSampleExp.py -r $i -a MILP
done
