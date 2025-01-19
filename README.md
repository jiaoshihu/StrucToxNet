# StrucToxNet

# 1 Description

StrucToxNet is a novel computational framework specifically designed to predict peptide toxicity by integrating both sequential and structural information. The model leverages a pre-trained protein language model, ProtT5, to extract rich sequential features from peptide sequences, and incorporates a 3D equivariant graph neural network (EGNN) using structural data predicted by ESMFold to capture spatial characteristics. By combining these complementary feature sets, StrucToxNet significantly enhances predictive accuracy and generalization compared to existing sequence-based methods. Comprehensive evaluations demonstrate that StrucToxNet achieves robust performance and outperforms state-of-the-art tools on independent datasets. This framework has the potential to accelerate the development of safer peptide-based therapeutics by enabling reliable and efficient toxicity screening.


## Installation Guide

### Step 1: Clone the Repository

Run the following commands in your terminal:
- `git clone https://github.com/jiaoshihu/StrucToxNet.git`
- `cd StrucToxNet`

### Step 2: Set Up the Environment with Conda

1. Use the `environment.yml` file to create the environment by running:
   - `conda env create -f environment.yml`
2. Activate the environment:
   - `conda activate test1`

### Step 4: Preprocessing

- `cd Preprocessing`
- Place the PDB files in the directory ./inputs/pdb/.
- Place the corresponding FASTA files in the directory ./inputs/.
- Run the preprocessing script to process the data.
- python 1_preprocess.py -i test_sequences.txt
- python 2_get_features.py -i test_sequences.txt
- python 3_get_plm.py -i test_sequences.txt
- python 4_feature_all.py
