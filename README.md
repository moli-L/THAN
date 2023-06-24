# THAN
Code for "[Transformer-based Representation Learning on Temporal Heterogeneous Graphs](https://link.springer.com/chapter/10.1007/978-3-031-25198-6_29)" and "[Memory-enhanced transformer for representation learning on temporal heterogeneous graphs](https://link.springer.com/article/10.1007/s41019-023-00207-w)".


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

Download dataset from:
- [https://grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k)
- [http://snap.stanford.edu/data/higgs-twitter.html](http://snap.stanford.edu/data/higgs-twitter.html)
- [http://snap.stanford.edu/data/sx-mathoverflow.html](http://snap.stanford.edu/data/sx-mathoverflow.html)

### Preprocess the data

If there is no data in `processed` folder, run
```python
python process_data.py
```


# Training

### THAN 

temporal link prediction task
```python
# on movielens dataset
python learn_edge_all.py -d movielens --n_layer 2 --bs 500 --n_degree 8 --n_epoch 30 --lr 1e-3

# on twitter dataset
python learn_edge_all.py -d twitter --n_layer 2 --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-4

# on mathoverflow dataset
python learn_edge_all.py -d mathoverflow --n_layer 2 --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-3
```

inductive experiment
```python
# on movielens dataset
python learn_edge_new.py -d movielens --n_layer 2 --bs 500 --n_degree 8 --n_epoch 30 --lr 1e-3

# on twitter dataset
python learn_edge_new.py -d twitter --n_layer 2 --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-4

# on mathoverflow dataset
python learn_edge_new.py -d mathoverflow --n_layer 2 --bs 800 --n_degree 10 --n_epoch 20 --lr 1e-3
```

If the memory size of your GPU is less than 24GB, please reduce the batch size.


### THAN with Memory 

temporal link prediction task
```python
# on movielens dataset
python learn_edge_all.py -d movielens --prefix THAN-mem --use_memory --n_degree 8 --n_epoch 30 --lr 1e-3

# on twitter dataset
python learn_edge_all.py -d twitter --prefix THAN-mem --use_memory --n_degree 10 --n_epoch 20 --lr 1e-4

# on mathoverflow dataset
python learn_edge_all.py -d mathoverflow --prefix THAN-mem --use_memory --n_degree 10 --n_epoch 20 --lr 1e-3
```

inductive experiment
```python
# on movielens dataset
python learn_edge_new.py -d movielens --prefix THAN-mem --use_memory --n_degree 8 --n_epoch 30 --lr 1e-3

# on twitter dataset
python learn_edge_new.py -d twitter --prefix THAN-mem --use_memory --n_degree 10 --n_epoch 20 --lr 1e-4

# on mathoverflow dataset
python learn_edge_new.py -d mathoverflow --prefix THAN-mem --use_memory --n_degree 10 --n_epoch 20 --lr 1e-3

