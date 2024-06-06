# Enviroment

python >=3.9\
torch==2.2.1\
transformers==4.39.3

For detailed environment setup, please refer to the requirements.txt.

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

```sh
python LLM_model.py <method> --sentence <sentence> [--check <check>] [--train <train>] [--test <test>] [--SEED <seed>]

Basic Example
python LLM_model.py window --sentence 5

Example with All Arguments
python LLM_model.py memory --sentence 5 --check 5 --train 1 --test 1 --SEED 123

Note: Running the LLM model requires a significant amount of memory and time, so please ensure you have a sufficient environment. If your environment is not sufficient, it is recommended to set train and test to 0 and review the experimental results data we have provided.
