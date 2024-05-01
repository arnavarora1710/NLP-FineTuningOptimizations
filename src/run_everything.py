import os, subprocess

# Change configs here!

# USAGE FOR TRAIN:
# python3 train.py <problem_num> <train_data_path> <test_data_path> <num_epochs> <learning_rate>*

# USAGE FOR SOUPER:
# python3 souper.py <problem_num> <test_data_path> <weights_folder> <quantize_or_not> <soup_type>

# Create result directories if they don't exist
result_directories = [
    "../results/sql",
    "../results/crisis",
    "../results/stock"
]

for directory in result_directories:
    os.makedirs(directory, exist_ok=True)

configs = [
    "python3 ../src/train.py 1 ../data/SQLInjections/sqli.csv ../data/SQLInjections/sqliv2.csv 15 1e-5 2e-5 3e-5",
    "python3 ../src/train.py 2 ../data/CrisisNLP/train.csv ../data/CrisisNLP/test.csv 15 1e-5 2e-5 3e-5",
    "python3 ../src/train.py 3 ../data/StockPredictions/train.csv ../data/StockPredictions/test.csv 15 1e-5 2e-5 3e-5",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ False mean > ../results/sql/sql_mean.txt",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ True mean > ../results/sql/sql_mean_quantized.txt",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ False median > ../results/sql/sql_median.txt",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ True median > ../results/sql/sql_median_quantized.txt",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ False weight > ../results/sql/sql_weighted_mean.txt",
    "python3 ../src/souper.py 1 ../data/SQLInjections/sqliv2.csv ../src/sql_weights/ True weight > ../results/sql/sql_weighted_mean_quantized.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ False mean > ../results/crisis/crisis_mean.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ True mean > ../results/crisis/crisis_mean_quantized.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ False median > ../results/crisis/crisis_median.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ True median > ../results/crisis/crisis_median_quantized.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ False weight > ../results/crisis/crisis_weighted_mean.txt",
    "python3 ../src/souper.py 2 ../data/CrisisNLP/test.csv ../src/crisis_weights/ True weight > ../results/crisis/crisis_weighted_mean_quantized.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ False mean > ../results/stock/stock_mean.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ True mean > ../results/stock/stock_mean_quantized.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ False median > ../results/stock/stock_median.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ True median > ../results/stock/stock_median_quantized.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ False weight > ../results/stock/stock_weighted_mean.txt",
    "python3 ../src/souper.py 3 ../data/StockPredictions/test.csv ../src/stock_weights/ True weight > ../results/stock/stock_weighted_mean_quantized.txt"
]

for config in configs:
    subprocess.run(config, shell=True)