move gmon.out bin
cd bin
gprof main.exe gmon.out >  analysis.txt
python gprof2dot.py -n0.5 -s analysis.txt > analysis.dot
dot -Tpng analysis.dot -Gcharset=latin1 -o analysis.png
cd ..