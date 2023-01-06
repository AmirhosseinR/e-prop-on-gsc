# E-prop: the code repository

Here is the tensorflow simulation for GSC: `eprop_LSNN.py`

The c simulation can be found in the following folder:

`c\`



# Run eprop_LSNN.py

Install CONDA.  Conda cheat sheet can be found: https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf

The conda config file is in `eprop-GSC.yml`, this is the package version in my PC. Just pay attention to main library version like: python, tensorflow, librosa, scipy, pyaudio.

- Make sure there is no space, dash (-) or special character (#, @, ...) in the file path (otherwise you get python library error).


# Result

The result for TensorFlow v2.4.1 and CUDA v11.1.1 on NVIDIA GA100 [A100 SXM4 40GB]:

It takes about 3H for each run.

| Run         | Epoch       | Train acc. (%) | **Best** Valid acc. (%) | Test acc. (%) |
|-------------|-------------|----------------|-------------------------|---------------|
| 1           | 28          | 95.99          | 91.04                   | 91.4          |
| 2           | 21          | 95.67          | 90.98                   | 90.9          |
| 3           | 15          | 95.29          | 91.11                   | 91.1          |
| 4           | 28          | 96.04          | 91.11                   | 91.4          |
| 5           | 23          | 95.72          | 90.84                   | 91.2          |
| 6           | 25          | 95.9           | 90.8                    | 91.3          |
| 7           | 24          | 95.89          | 90.84                   | 91.4          |
| 8           | 26          | 96.07          | 90.94                   | 91.4          |
| 9           | 27          | 95.86          | 91                      | 91.4          |
| 10          | 26          | 95.81          | 91.15                   | 91.7          |
| 11          | 28          | 95.86          | 90.66                   | 91.4          |
| 12          | 21          | 95.81          | 91.06                   | 91            |
| **Average** | 24.33       | 95.83          | 90.96                   | 91.3          |

