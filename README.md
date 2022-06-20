# THAN
Code for "Transformer-based Representation Learning on Temporal Heterogeneous Graphs".


# Requirements
Dependencies (with python >= 3.9):
```
numpy==1.20.3
pandas==1.3.3
torch==1.9.1
scikit_learn==0.24.2
```

run
```
pip install -r requirements.txt
```


# Preprocessing

### Dataset

Create a folder `data` to store source data files.

### Preprocess the data

If there is no data in `processed` folder, run
```python
python process_data.py
```


# Training

temporal link prediction task
```python
# on movielens dataset
python learn_edge_all.py -d movielens --bs 500 --n_degree 8 --n_epoch 30 --lr 1e-3 --gpu 0

# on twitter dataset
python learn_edge_all.py -d twitter --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-4 --gpu 0
```

inductive experiment
```python
python learn_edge_new.py -d movielens --bs 500 --n_degree 8 --n_epoch 30 --lr 1e-3 --gpu 0

# on twitter dataset
python learn_edge_new.py -d twitter --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-4 --gpu 0
```

If the memory size of your GPU is less than 24GB, please reduce the batch size.

