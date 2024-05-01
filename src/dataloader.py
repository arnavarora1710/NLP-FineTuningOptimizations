import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

maps = {
    'injured_or_dead_people': 0,
    'missing_trapped_or_found_people': 1,
    'displaced_people_and_evacuations': 2,
    'infrastructure_and_utilities_damage': 3,
    'donation_needs_or_offers_or_volunteering_services': 4,
    'caution_and_advice': 5,
    'sympathy_and_emotional_support': 6,
    'other_useful_information': 7,
    'not_related_or_irrelevant': 8
}

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
        self.data = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip', header=0)
        self.data = self.data[self.data['label'].notna()]
        self.data['label'] = self.data['label'].apply(lambda x: maps[x])
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['tweet_text'])
        label = int(self.data.iloc[idx]['label'])
        one_hot_label = [1.0 if i == label else 0.0 for i in range(9)]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(one_hot_label)
        }

def get_crisis_dataloader(data_path, tokenizer, max_length, batch_size):
    dataset = CrisisDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

class StockDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip', header=0)
        self.data.fillna('', inplace=True)
        self.data['Concatenated_Top'] = self.data.apply(lambda row: ';'.join(row[2:]), axis=1)
        self.data = self.data[self.data['Label'].notna()]
        self.data = self.data[pd.to_numeric(self.data['Label'], errors='coerce').notnull()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['Concatenated_Top'])
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
