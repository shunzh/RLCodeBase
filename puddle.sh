for i in `seq 0 9`
do
  for j in 1 3 5 7 9
do
  python puddleExp.py -q full -r $i -a AS -m $j
done
done
