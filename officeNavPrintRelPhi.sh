for r in `seq 0 19`
do
  python officeNavigation.py -n 5 -r $r
  python officeNavigation.py -n 10 -r $r
  python officeNavigation.py -n 15 -r $r
done

