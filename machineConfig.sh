for i in `seq 0 399`;
do
  python machineConfigExp.py 0.05
  python machineConfigExp.py 0.1
  python machineConfigExp.py 0.2
  python machineConfigExp.py 0.3
  python machineConfigExp.py 0.5
done
