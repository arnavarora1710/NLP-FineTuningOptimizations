import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
torch.manual_seed(42)

train_data_path = "../data/SQLInjections/sqli.csv"
test_data_path = "../data/SQLInjections/sqliv2.csv"

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='utf-16', 
        on_bad_lines='skip', header=None, names=['Sentence', 'Label'])
        self.data = self.data[self.data['Label'].notna()]
        self.data = self.data[pd.to_numeric(self.data['Label'], errors='coerce').notnull()]
        print(self.data['Label'].value_counts())
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['Sentence'])
        label = int(self.data.iloc[idx]['Label'])
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = CustomDataset(train_data_path, tokenizer, max_length=64)
test_dataset = CustomDataset(test_data_path, tokenizer, max_length=64)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=1e-5)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    sum = 0
    cnt = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        sum += loss.item()
        cnt += 1
        print(f'Loss: {loss.item()}')
    print()
    print(f"Average Epoch Loss: {sum/cnt}")
    print()

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
num_correct = 0
total_predictions = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        
        num_correct += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

accuracy = num_correct / total_predictions
print(f"Accuracy on the test set: {accuracy}")