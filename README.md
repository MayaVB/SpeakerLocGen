# BIUREVgen
Python library that creates the BIUREV and BIUREV-N datasets used in the 2021 INTERSPEECH paper ["Scene-Agnostic Multi-Microphone Speech Dereverberation"](https://arxiv.org/pdf/2010.11875.pdf).

It simulates reverberant speech under different acoustic conditions recorded by an eight-microphone array. 

The simulated speech is obtained by convoloving room impulse responses (RIRs) generated by the [gpuRIR package](https://github.com/DavidDiazGuerra/gpuRIR) with clean speech signals taken from the [REVERB challenge](http://reverb2014.dereverberation.com/).
Each dataset consists of training, validation and evauation data.
For each dataset each split comprises the following scenarios:
1. Training: Random.
2. Validation: Near, Far.
3. Test: Near, Far, Random and Winning Ticket

## Prerequisites
#### REVERB
You must have the clean training, validation and evaluation files of REVERB to use our code.
Note that in the REVERB dataset, the clean WAV files are stored in nested directories, e.g. for the training data:  
REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/**primary_microphone/si_tr**/c0a  
REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/**primary_microphone/si_tr**/c0b  
...<br/>
In order for the code to work properly, please remove the **intermediate nested directories** and put all the end directories as the immediate children of cln_train, i.e.  
REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/c0a  
REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/ c0b  
...  

Repeat the procedure, such that for validation data the structure is:  
REVERB/SimData/REVERB_WSJCAM0_dt/data/cln_test/c31  
REVERB/SimData/REVERB_WSJCAM0_dt/data/cln_test/c34  
...  

and for evaluation data the structure is:  
REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test/c30  
REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test/c32  
...  

#### Python
This code was designed with Python 3. Backwards compatability to Python 2 therefore cannot be guaranteed.  
Note that the provided code should be run on a GPU.
The following packages are required for running the code:
1. [gpuRIR package](https://github.com/DavidDiazGuerra/gpuRIR).
2. soundfile.

## Files
- `rev_speech_gen` - generates the entire dataset.
- `scene_gen` - simulates an acoustic scenario, i.e. room dimesions, speaker and microphones placements and reverberation time.
- `rir_gen` - simulates a RIR given an acoustic scenario.

## Usage
In `rev_speech_gen`, update the variable `clean_speech_dir` to store the directory in which the clean speech files can be found.  
This must be done for the training-validation-test if-else.  
You can subsequently run the file from the command line, for example:
```
python rev_speech_gen.py --split train --dataset BIUREV
```

## Citation
If you use this code/datasets for your research paper, please kindly cite:
```
@INPROCEEDINGS{dss_dereverb,  
  author={Yochai Yemini, Ethan Fetaya, Haggai Marron and Sharon Gannot},  
  booktitle={Proc. of INTERSPEECH},  
  title={Scene-Agnostic Multi-Microphone Speech Dereverberation},  
  year={2021},  
  volume={},  
  number={},  
  pages={},  
}
```
