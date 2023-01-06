# e-prop on GSC

This repository contains similar codes which was used by authors of `E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware` paper [1].

The code used in the paper was written and tested on TensorFlow version 1 and then it was modified to TensorFlow version 2.

The repository includes five sub-folders:

- preprocess: Use Mel-Frequency Campestral Coefficients (MFCCs) to extract features and save it to files. One should run it ones before training to generate dataset. 
  - We used the codes from `https://github.com/douglas125/SpeechCmdRecognition` repository and modified a few lines to satisfy our requirements.
- tf2: e-prop code in TensorFlow version 2 for training Google Speech Command (GSC) Dataset.
  - We used codes from e-prop official github repository `https://github.com/IGITUGraz/eligibility_propagation` and adopted it to TensorFlow version 2 and GSC.
- c: c code for e-prop, does not depend on any library and can be run on CPUs, microcontrollers (bare-metal) (some minor changes is needed to feed dataset to microcontrollers in an appropriate way) and ... .
  - The codes used in the paper is based on this folder, but there are two main difference, the code used in the paper:
    - we rearrange some part to make it online 
    - implemented it by using 12 cores, added synchronization process and used SpiNNaker 2 timer, comms unit and DMA hardware peripherals.
  - Also the SpiNNaker 2 c code will be made available upon request (It only works on SpiNNaker 2!).
- files: contains initial weights and preprocessed dataset
- result: contains the result of tf2 folder (accuracy, loss of different dataset and trained weight for best validation set)

The preprocess dataset (after running codes in the preprocess folder) and initial random weights for c code (after running tf2 folder) should be available in the `files` folder (same hierarchy as c, tf2 and preprocess folders). A compressed `files` folder (files.zip) that contains a small portion of preprocess dataset and also initial random weights are available in the git repository and can be extracted and use for demonstration purpose.

[1] E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware,
A Rostami, B Vogginger, Y Yexin, C G Mayr
[Frontiers in Neuroscience](https://www.frontiersin.org/articles/10.3389/fnins.2022.1018006/full), November 2022
## Citation

If you find the codes useful, please cite our work:

```
@ARTICLE{10.3389/fnins.2022.1018006,
AUTHOR={Rostami, Amirhossein and Vogginger, Bernhard and Yan, Yexin and Mayr, Christian G.},    
TITLE={E-prop on SpiNNaker 2: Exploring online learning in spiking RNNs on neuromorphic hardware},
JOURNAL={Frontiers in Neuroscience},
VOLUME={16},
YEAR={2022},
URL={https://www.frontiersin.org/articles/10.3389/fnins.2022.1018006},
DOI={10.3389/fnins.2022.1018006},
ISSN={1662-453X},
}
```
