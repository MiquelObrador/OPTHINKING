#Open the json file
import json

def read_data(filename):
    with open(filename, 'r', encoding="utf8") as file:
        data = json.load(file)
    return data

import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Dataset_en(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.data = read_data(path)
        self.tokenizer = tokenizer
        self.max_len = 512
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['category']
        if label == "CONSPIRACY":
            label = 1
        else:
            label = 0
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        item = {key: inputs[key].squeeze(0) for key in inputs}
        item['labels'] = torch.tensor(label)
        
        return item
    
dataset = Dataset_en("dataset_en_train.json", tokenizer)

print(len(dataset))

#Train test split the dataset
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

print(len(train_data))
print(len(test_data))

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

print(model)
    
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef

results = {}

def train(model, train_loader, num_epochs, optimizer):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        predictions = []
        ground_truth = []
        
        for batch in tqdm(train_loader):
            inputs = {key: batch[key].to(device) for key in batch if key != 'labels'}
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            predictions.extend(torch.argmax(outputs.logits, axis=1).tolist())
            ground_truth.extend(labels.tolist())
            
        avg_train_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {avg_train_loss:.4f} - Training F1 score: {f1_score(ground_truth, predictions)} - Training MCC: {matthews_corrcoef(ground_truth, predictions)}")
        
        f1_sco, MCC, avg_val_loss = evaluate(model, test_loader)
        
        print(f"Validation loss: {avg_val_loss:.4f} - Validation F1 score: {f1_sco} - Validation MCC: {MCC}")
        
        results[epoch] = {'val_loss': avg_val_loss, 'f1_score': f1_sco, 'MCC': MCC}

def evaluate(model, test_loader):
    model.eval()
    total_val_loss = 0
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = {key: batch[key].to(device) for key in batch if key != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            total_val_loss += loss.item()
            
            predictions.extend(torch.argmax(outputs.logits, axis=1).tolist())
            ground_truth.extend(labels.tolist())
            
    avg_val_loss = total_val_loss / len(test_loader)
    
    return f1_score(ground_truth, predictions), matthews_corrcoef(ground_truth, predictions), avg_val_loss

train(model, train_loader, num_epochs, optimizer)

#Save the results
with open("results_base_en.json", "w") as file:
    json.dump(results, file)

#Save the model
model.save_pretrained("model_base_en")