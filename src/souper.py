import sys
sys.path.append("../")
import time

import numpy as np
from sklearn.metrics import f1_score

import torch
from dataloader import get_sql_dataloader, get_crisis_dataloader, get_stock_dataloader
from transformers import AutoTokenizer, BertForSequenceClassification 

from trainv1 import test_model

def avg_soup_models(weights):
    avg_weights = {}
    for key in weights[0].keys():
        curr_weights = [w[key].to(torch.float32) for w in weights]
        curr_stack = torch.stack(curr_weights)
        avg_weights[key] = torch.mean(curr_stack, dim=0)
        
    return avg_weights


def test_weights(model, weights, test_dataset, test_dataloader, quantize=False):
    model.load_state_dict(weights)
    
    if quantize:
        model.to(torch.float16)
    
    start_time = time.time()
    labels, predictions = test_model(model, test_dataset, test_dataloader)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
    
    f1 = f1_score(labels, predictions)
    accuracy = np.mean(labels == predictions)
    
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print("\n\n\n\n")
    

if __name__ == "__main__":
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(sys.argv) < 4:
        print("Usage: python souper.py <problem_num> <test_data_path> <weights_folder>")
        sys.exit(1)

    problem_num = int(sys.argv[1]) - 1
    test_data_path = sys.argv[2]
    weights_folder = sys.argv[3]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    dataloader_funs = [get_sql_dataloader, get_crisis_dataloader, get_stock_dataloader]
    test_dataset, test_loader = dataloader_funs[problem_num](test_data_path, tokenizer, max_length=64, batch_size=BATCH_SIZE)
    
    weight_files = ["1e-05.pth", "5e-05.pth", "0.00025.pth"]
    weight_files = [weights_folder + w for w in weight_files]
    weights = [torch.load(w) for w in weight_files]
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    
    for weight, weight_file in zip(weights, weight_files):
        print(f"Testing {weight_file}")
        test_weights(model, weight, test_dataset, test_loader)
    
    avg_weights = avg_soup_models(weights)
    
    print("Testing averaged quantized weights")
    test_weights(model, avg_weights, test_dataset, test_loader, True)
