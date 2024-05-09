# Foldseek_Tweaker 🐰
The Foldseek_Tweaker can adjust the number of alphabet sizes for Foldseek and is helpful for beginners to understand the Foldseek code of training.

## Introduction
[*Foldseek*](https://github.com/steineggerlab/foldseek) is a breakthrough that transforms the structure of proteins into sequences, greatly accelerating the sequence alignment! 

However, it has chosen 20 as its alphabet size for the trade-off between effectiveness and expressiveness. So, imagine we want to pursue one side. Can we adjust this size? Let's try!

## Installation
Here are some configurations from our device for reference:

- CUDA == 11.7
- Python == 3.9
- torch == 1.13.0+cu117
- biopython == 1.83

Please be mindful of version compatibility during your actual setup.

## Quick Start
- *Bash Version*: `sudo bash Encoder.sh tmp/pdb 20`
	- *20*: Alphabet Size You Want
	- *tmp/pdb*: Data Directory
 
The result is named as *seqs.csv* and is saved in the *tmp_20* folder. We have trained 40, 80, 128, and 512 for you to choose from.

## Train Your Own Foldseek
- *Bash Version*: `sudo bash learnAlphabet.sh 20 100 data/pdbs_train.txt data/pdbs_val.txt tmp_20/`
	- *20*: Alphabet Size You Want
	- *100*: Seed for Model Selection
  - *data/pdbs_train.txt*: Training Data File (See Data Preprocess)
  - *data/pdbs_val*: Validation Data File (See Data Preprocess)
  - *tmp_20/*: Output Directory
- *Local*: Training Step by Step at one seed
	- Train `train_vqvae_local.py`
	- Encode `encode_pdbs_local.py`
	- Evalue `create_submat2.py`

## Data Preprocess
  1. (Optional) Please make sure **your protein data** has been put in **tmp/pdb**, or you can run this code on the terminal to test example data.
      ```PowerShell
      if [ ! -d tmp/pdb ]; then
        curl https://wwwuser.gwdg.de/~compbiol/foldseek/scop40pdb.tar.gz | tar -xz -C tmp
      fi
      ```
  2. Compile ssw_test on your terminal
     ```PowerShell
     if [ ! -f tmp/ssw_test ]; then
        git clone --depth 1 https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library tmp/ssw
        (cd tmp/ssw/src && make)
        cp tmp/ssw/src/ssw_test tmp/ssw_test
     fi
     ```
  3. Create Training Data `create_vqvae_training_data_local.py`
