# StrucToxNet

# 1 Description

StrucToxNet is a novel computational framework specifically designed to predict peptide toxicity by integrating both sequential and structural information. The model leverages a pre-trained protein language model, ProtT5, to extract rich sequential features from peptide sequences, and incorporates a 3D equivariant graph neural network (EGNN) using structural data predicted by ESMFold to capture spatial characteristics. By combining these complementary feature sets, StrucToxNet significantly enhances predictive accuracy and generalization compared to existing sequence-based methods. Comprehensive evaluations demonstrate that StrucToxNet achieves robust performance and outperforms state-of-the-art tools on independent datasets. This framework has the potential to accelerate the development of safer peptide-based therapeutics by enabling reliable and efficient toxicity screening.


# 2 Requirements

Before running, please make sure the following packages are installed in Python environment:

python==3.8

pytorch==1.13.1

numpy==1.24.2

pandas==1.5.3



# 3 Running

Changing working dir to ATGPred-main, and then running the following command:

python main.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
