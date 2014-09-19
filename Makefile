clear:
	rm tg_*result q_*result

exp1:
	python blocksworld.py -a tg -k 1000 -q
	python blocksworld.py -a tg -k 1000 --k1 5 -q

exp2:
	python blocksworld.py -a q -k 1000 -q
	python blocksworld.py -a q -k 1000 --k1 5 -q

try:
	python blocksworld.py -a tg -k 10

show1:
	ls tg_*result q_*result | xargs python process.py
	gnuplot exp1.gplot > exp1.ps
	gv exp1.ps
show2:
	ls tg_*result q_*result | xargs python process.py
	gnuplot exp2.gplot > exp2.ps
	gv exp2.ps
show3:
	ls tg_*result q_*result | xargs python process.py
	gnuplot exp3.gplot > exp3.ps
	gv exp3.ps
