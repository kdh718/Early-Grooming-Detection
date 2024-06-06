import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


import ast
import torch
import faiss
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as ds
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report, fbeta_score, recall_score, precision_score
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

class GroomingDataset(ds):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data
    
    def __len__(self):
        return len(self.labels)
    
class Memory:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.memory = []
    
    def add_to_memory(self, text, embedding):
        self.memory.append(text)
        self.index.add(np.array([embedding]))

    def search_memory(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        results = [self.memory[i] for i in I[0]]
        return results

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
def get_embedding(text):
    return embedder.encode([text])[0]
    
def predict(temp_set, model, threshold=0.7):
    test_dataloader = DataLoader(temp_set, batch_size=1, shuffle=False)
    
    model.eval()
    for data in test_dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids = data['input_ids'].to(device),
                attention_mask = data['attention_mask'].to(device),
                token_type_ids = data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy() 
        logits = logits.detach().cpu().numpy()
        if prob[0][1]>=threshold:
            return prob[0][1], 1
        else:
            return prob[0][1], 0     
        
def penalty_metric(d, n):
    p = np.log(3)/(d/2 - 1) if d/2 > 1 else 0.5
    return -1 + 2/(1+np.exp(-p * (n-1)))

def speed_metric(n):
    return 1 - np.median(n)

def main():
    parser = argparse.ArgumentParser(description="Early grooming detection")
    
    parser.add_argument('model_name', choices=['bert', 'koelectra','roberta'], help='chooes model among bert/koelectra/roberta')
    parser.add_argument('method', choices=['window', 'memory'], help="Specify the method to use: 'window' or 'memory'")
    parser.add_argument('--threshold', type=float, default=0.7,help="Specify the softmax threshold", required=False)
    parser.add_argument('--sentence', type=int, default=3,help="the number of concated sentences", required=True)
    parser.add_argument('--train', type=int, default=0,help="1=train, 0=no train, if no train, use same model used in paper", required=False)
    parser.add_argument('--test', type=int, default=0,help="1=test, 0=no test, if no test, use same data used in paper, test means early detection", required=False)
    parser.add_argument('--SEED', type=int, default=42, help="set seed(default seed=42)", required=False)
    
    args = parser.parse_args()
    SEED = args.SEED
    seed_everything(SEED)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_name == 'bert':
        MODEL_NAME = 'klue/bert-base'
    elif args.model_name == 'koelectra':
        MODEL_NAME = 'monologg/koelectra-small-v3-discriminator'
    else:
        MODEL_NAME = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if args.train:
        df = pd.read_csv('./datasets/Koran_SNS_datasets.csv', index_col=0)
        df_normal = df[df['labels']==0].sample(5000, random_state=SEED)
        df_sexual = df[df['labels']==1].sample(10000, random_state=SEED)
        df = pd.concat([df_normal,df_sexual])
        df = df.sort_values(by='cid')
        df = df.reset_index(drop=True)
        train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'], shuffle=True, random_state=SEED)
        train_data = Dataset.from_pandas(train_df)
        test_data = Dataset.from_pandas(test_df)
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2)

        train_encoding = tokenizer(
        train_data['text'],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=64,
        add_special_tokens=True
        )

        test_encoding = tokenizer(
            test_data['text'],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64,
            add_special_tokens=True
        )
        
        trainset = GroomingDataset(train_encoding, train_data['labels'])
        testset = GroomingDataset(test_encoding, test_data['labels'])
        
        training_args = TrainingArguments(
            output_dir='./models',
            logging_dir='./results',
            num_train_epochs=50,
            per_device_train_batch_size=24,
            per_device_eval_batch_size=24,
            logging_steps=100,
            evaluation_strategy='no',
            save_strategy='no',
            learning_rate=1e-5
        )
        model.to(device)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=None
        )
        print('Train start!')
        trainer.train()
        model.save_pretrained(f'models/result_model_folder/{args.model_name}_{current_time}.model')
        print('Train done!')
        
        test_loader = DataLoader(
            testset,
            batch_size=16,
            shuffle=False
        )
        preds, probs = [], []
        truths = []
        model.eval()
        
        for data in test_loader:
            with torch.no_grad():
                outputs = model(
                    input_ids = data['input_ids'].to(device),
                    attention_mask = data['attention_mask'].to(device),
                    token_type_ids = data['token_type_ids'].to(device)
                )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)
            
            truths.append(data['labels'])
            preds.append(result)
            probs.append(prob)
        preds = np.concatenate(preds).tolist()
        probs = np.concatenate(probs).tolist()
        truths = np.concatenate(truths).tolist()
        
        print('Train result f1 : %.2f, acc : %.2f'%(f1_score(truths, preds), accuracy_score(truths, preds)))
        
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f'./models/{args.model_name}.model',num_labels=2).to(device)
    
    df_test = pd.read_csv('./datasets/data.csv', index_col=0)
    df_test['text'] = [ast.literal_eval(dq_txt) for dq_txt in df_test['text']]
    df_test['labels'] = [1 if lab>0 and lab<5 else 0 for lab in df_test['labels']]
    df_test['summary'] = ['성적' if lab>0 and lab<5 else '일반' for lab in df_test['labels']]
    df_test = df_test[df_test['text'].apply(len).gt(2)]
    df_test['length'] = df_test['text'].apply(len)
    
    pred_list, line_list = [], []
    if args.test:
        pbar = tqdm(total = len(df_test))
        if args.method == 'window':
            for dq in df_test.itertuples():
                FLAG = True
                for idx in range(1, dq.length):
                    start = idx-args.sentence if idx>args.sentence else 0
                    end = idx
                    premise = ' [SEP] '.join(dq.text[start:end])
                    temp_df = pd.DataFrame({"cid":'C_000001',"text":premise, "labels":0}, index=[0])
                    temp_data = Dataset.from_pandas(temp_df)
                    temp_encoding = tokenizer(
                        temp_data['text'],
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=64,
                        add_special_tokens=True
                        )
                    temp_set = GroomingDataset(temp_encoding,temp_data["labels"])
                    prob, result = predict(temp_set, model, args.threshold)
                    if result==1:
                        pred_list.append(1)
                        line_list.append(end)
                        FLAG = False
                        break
                if FLAG:
                    pred_list.append(0)
                    line_list.append(idx+1)
                pbar.update(1)
        else:
            model.eval()
            for dq in df_test.itertuples():
                FLAG = True
                memory = Memory(dim=embedder.get_sentence_embedding_dimension())
                for idx in range(dq.length):
                    input_text = dq.text[idx]
                    if idx:
                        query_embedding = get_embedding(input_text)
                        search_results = memory.search_memory(query_embedding, top_k=args.sentence)
                        premise_list = [input_text]
                        premise_list = premise_list + list(set(search_results))
                        premise = " [SEP] ".join(premise_list)
                    else:
                        premise = input_text
                    temp_df = pd.DataFrame({"cid":'C_000001',"text":premise, "labels":0}, index=[0])
                    temp_data = Dataset.from_pandas(temp_df)
                    temp_encoding = tokenizer(
                        temp_data['text'],
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=64,
                        add_special_tokens=True
                    )
                    temp_set = GroomingDataset(temp_encoding,temp_data["labels"])
                    prob, result = predict(temp_set, model, args.threshold) #윈도우 크기만큼의 문장, 사용할 모델, 임계치
                    if result==1:
                        pred_list.append(1)
                        line_list.append(idx+1)
                        FLAG = False
                        break
                    embedding = get_embedding(input_text)
                    memory.add_to_memory(input_text, embedding)
                if FLAG:
                    pred_list.append(0)
                    line_list.append(idx+1)
                del memory
                pbar.update(1)
        pbar.close()
        df_test['pred'] = pred_list
        df_test['detected_line'] = line_list    
    else:
        df_test = pd.read_csv(f'./datasets/ExperimentDatasets/{args.model_name}_{args.method}.csv')
                
    print('Early detection f1 : %.2f, f3 : %.2f, acc : %.2f'%(f1_score(df_test['labels'], df_test['pred']),fbeta_score(df_test['labels'], df_test['pred'], beta=3),accuracy_score(df_test['labels'], df_test['pred'])))
    print('Early detection recall : %.2f, precision : %.2f'%(recall_score(df_test['labels'], df_test['pred']),precision_score(df_test['labels'], df_test['pred'])))
        
    penalty_array = np.array([penalty_metric(data.length,data.detected_line) for data in df_test.itertuples() if data.labels and data.pred])
    speed = speed_metric(penalty_array)
    latency_f1 = speed * f1_score(df_test['labels'],df_test['pred'])
    print('speed : %.2f, latency_f1 : %.2f'%(speed, latency_f1))
    
    df_test.to_csv(f'datasets/result_csv_folder/{args.model_name}_{args.method}_{current_time}.csv')
    
    df_EDR = pd.read_csv('./datasets/sentence_data_for_EDR.csv', index_col=0)
    EDR_list = []
    for data in df_test.itertuples():
        if data.labels and data.pred:
            temp = df_EDR[df_EDR['cid']==data.cid].reset_index(drop=True)
            first = temp[temp['labels']==1].iloc[0]['speed']
            detected = temp['speed'][data.detected_line - 1]
            EDR_list.append(round(detected/first,2))
    EDR_list = np.array(EDR_list)
    
    print('EDR min : %.2f, max : %.2f, mean : %.2f'%(EDR_list.min(),EDR_list.max(),EDR_list.mean()))
    

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if __name__ == "__main__":
    main()