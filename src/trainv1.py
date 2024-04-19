import sys
sys.path.append("../")
import time

import numpy as np
from sklearn.metrics import f1_score

import torch
torch.manual_seed(42)
from datasets import get_sql_dataloader, get_crisis_data_loader, get_stock_data_loader
from transformers import AutoTokenizer, BertForSequenceClassification

def train_batch(model, batch, optimizer):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    c = 0
    for batch in train_loader:
        loss = train_batch(model, batch, optimizer)
        print(f"Batch Loss: {loss}")
        total_loss += loss
        c += 1
    return total_loss / c

def train_model(model, train_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

def test_batch(model, batch):
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    labels = batch['labels'].to(model.device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

    return labels, predictions

def test_model(model, test_dataset, test_loader):
    model.eval()
    labels = np.zeros(len(test_dataset))
    predictions = np.zeros(len(test_dataset))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            labels_batch, predictions_batch = test_batch(model, batch)
            start = i * len(labels_batch)
            end = start + len(labels_batch)
            labels[start:end] = labels_batch.cpu().numpy()
            predictions[start:end] = predictions_batch.cpu().numpy()
            
    return labels, predictions

if __name__ == "__main__":
    if (sys.argc < 3):
        print("Usage: python train.py <problem_num> <train_data_path> <test_data_path> <learning_rate>*")
        sys.exit(1)
    
    problem_num = sys.argv[1] - 1
    train_data_path = sys.argv[2]
    test_data_path = sys.argv[3]


    BATCH_SIZE = 8
    EPOCHS = 5
    NUM_MODELS = max(1, sys.argc - 3)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_funs = [get_sql_dataloader, get_crisis_data_loader, get_stock_data_loader]
    train_dataset, train_loader = dataloader_funs[problem_num](train_data_path, tokenizer, max_length=64, batch_size=BATCH_SIZE)
    test_dataset, test_loader = dataloader_funs[problem_num](test_data_path, tokenizer, max_length=64, batch_size=BATCH_SIZE)

    lrs = [float(lr) for lr in sys.argv[4:]]
    if len(lrs) == 0:
        lrs.append(1e-5)

    models = [BertForSequenceClassification.from_pretrained("bert-base-uncased") for _ in range(NUM_MODELS)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=lr) for model, lr in zip(models, lrs)]

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.to(device)
        train_model(model, train_loader, optimizer, EPOCHS)
        torch.save(model.state_dict(), f"sql_weights/{lrs[i]}.pth")

        # TODO: Vary this for different quantization levels
        model.to(torch.float16)
        
        start_time = time.time()
        labels, predictions = test_model(model, test_dataset, test_loader)
        end_time = time.time()

        print(f"Time taken: {end_time - start_time} seconds")
        
        f1 = f1_score(labels, predictions)
        accuracy = np.mean(labels == predictions)
        
        print(f"Learning Rate: {lrs[i]}")
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")
        print(f"\n\n\n\n")