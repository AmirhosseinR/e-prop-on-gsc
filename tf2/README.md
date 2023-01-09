# E-prop: TensorFlow v2 code repository

Here is the tensorflow simulation for GSC: `eprop_LSNN.py`

The c simulation can be found in the following folder:

`c\`



## Run eprop_LSNN.py

- The requirements packages are available in requirements.txt file.
- Make sure there is no space, dash (-) or special character (#, @, ...) in the python file path (otherwise you get python library error).
- The default code use e-prop with hard code (do not use TensorFlow auto-diff).
- Run the code:
  ```bash
  python3 eprop_LSNN.py
  ```


## Result

The result for TensorFlow v2.7.1 on NVIDIA A100-SXM4 GPU, it takes about 3 hours to complete:

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

