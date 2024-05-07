from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import torch
import numpy as np
from sklearn.metrics import f1_score


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def csv_split(index, category, category_values):
    X_train, X_val, y_train, y_val = train_test_split(index, 
                                                  category, 
                                                  test_size=0.15, 
                                                  random_state=42,
                                                  stratify=category_values)
    return X_train, X_val, y_train, y_val




def dataset_creation(df, target, text_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)   
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'][text_data].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'][text_data].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'][target].values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'][target].values)



    dataset_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
    dataset_val = TensorDataset(input_ids_val,attention_masks_val,labels_val)

    return dataset_train, dataset_val


def bertmodel(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels, output_attentions = False,output_hidden_states = False)
    return model

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def evaluate(dataloader_val,model):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b for b in batch)
        batch = tuple(b.to(device) if b is not None else None for b in batch)

        with torch.no_grad():        
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                     }

            outputs = model(**inputs)
            
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def training_model(dataset_train, dataset_val, num_labels):
    batch_size = 4
    dataloader_train = DataLoader(dataset=dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_val = DataLoader(dataset=dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size)


    model = bertmodel(num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(),lr=1e-5,eps=1e-8)
    epochs = 10
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=len(dataloader_train)*epochs)

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b for b in batch)
            batch = tuple(b.to(device) if b is not None else None for b in batch)
            inputs = {'input_ids': batch[0],'attention_mask': batch[1],'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total +=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))}) 

        tqdm.write('\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
    
        val_loss, predictions, true_vals = evaluate(dataloader_val,model=model)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (weighted): {val_f1}')

        torch.save(model, f'models/BERT.model')

        return val_loss, val_f1
