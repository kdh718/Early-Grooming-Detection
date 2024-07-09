# Enviroment

python >=3.9\
torch==2.2.1\
transformers==4.39.3

For detailed environment setup, please refer to the requirements.txt.

# Model Files

Due to file size limitations, the model file could not be uploaded to GitHub. Please download the file from the provided Google Drive link.

only model folders\
https://drive.google.com/drive/folders/1R_WgCQauVWqW3r0lSEKJR7ACosCO3DMs?usp=drive_link

total early detection folder\
https://drive.google.com/drive/folders/1ODvokkCCiYCH-68CFVvlYMFF9igpm74o?usp=drive_link


# How to Run the BERT-based Model

This script is used to run a BERT-based model. Use the following command to execute it.

## Command to Run

```sh
python BERT_based_model.py <model_name> <method> --sentence <sentence> [--threshold <threshold>] [--train <train>] [--test <test>] [--SEED <seed>]

Basic example
python BERT_based_model.py bert window --sentence 5

Example with All Arguments
python BERT_based_model.py koelectra window --threshold 0.7 --sentence 3 --train 0 --test 1 --SEED 42

# How to Run the LLM-based Model

This script is used to run an LLM-based model. Use the following command to execute it.

## Command to Run

python LLM_model.py <method> --sentence <sentence> [--check <check>] [--train <train>] [--test <test>] [--SEED <seed>]

Basic Example
python LLM_model.py window --sentence 5

Example with All Arguments
python LLM_model.py memory --sentence 5 --check 5 --train 1 --test 1 --SEED 123

Note: Running the LLM model requires a significant amount of memory and time, so please ensure you have a sufficient environment. If your environment is not sufficient, it is recommended to set train and test to 0 and review the experimental results data we have provided.

## Experiment Result
![experiment table](https://github.com/kdh718/Early-Grooming-Detection/assets/109021286/f48b1b83-3c34-4792-a203-bd39c77cbb2d)
