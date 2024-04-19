import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

class SQLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='utf-16', 
        on_bad_lines='skip', header=None, names=['Sentence', 'Label'])
        self.data = self.data[self.data['Label'].notna()]
        self.data = self.data[pd.to_numeric(self.data['Label'], errors='coerce').notnull()]
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
        
def get_sql_dataloader(data_path, tokenizer, max_length, batch_size):
    dataset = SQLDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

class CrisisDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='utf-16', 
        on_bad_lines='skip', header=None, names=['Sentence', 'Label'])
        self.data = self.data[self.data['Label'].notna()]
        self.data = self.data[pd.to_numeric(self.data['Label'], errors='coerce').notnull()]
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

def get_crisis_dataloader(data_path, tokenizer, max_length, batch_size):
    dataset = CrisisDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

class StockDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='utf-16', 
        on_bad_lines='skip', header=None, names=['Sentence', 'Label'])
        self.data = self.data[self.data['Label'].notna()]
        self.data = self.data[pd.to_numeric(self.data['Label'], errors='coerce').notnull()]
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

def get_stock_dataloader(data_path, tokenizer, max_length, batch_size):
    dataset = StockDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader