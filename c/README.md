
# Implementing of e-prop in ../tf2 folder in C

- First run the code in the preprocess folder to generate dataset.
- Make sure the preprocessed dataset is available in the appropriate path ("..\files") or change the dataset_path (dataset_path_trn_ftr, dataset_path_trn_trg, ...) in main.c.
- To generate random weights in `files` folder, one should run eprop_LSNN.py in tf2 folder at least one time.
- The codes was tested in Windows 10. Probably works in Linux (If there is not any conflict in Makefile and file path for Linux)

For profiling with gprof look at:
https://yzhong-cs.medium.com/profiling-with-gprof-64-bit-window-7-5e06ef614ba8

1) gprof main.exe gmon.out >  analysis.txt
2) python gprof2dot.py -n0.5 -s analysis.txt > analysis.dot
3) dot -Tpng analysis.dot -Gcharset=latin1 -o analysis.png
