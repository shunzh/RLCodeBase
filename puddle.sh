for i in `seq 0 20`;
do
  python puddleExp.py -r $i
done

for i in `seq 0 20`;
do
  python puddleExp.py -r $i -a H
done

for i in `seq 0 20`;
do
  python puddleExp.py -r $i -a RQ
done
