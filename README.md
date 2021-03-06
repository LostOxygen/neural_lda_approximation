# Neural LDA Approximation
Approximation of the latent dirichlet allocation via deep neural networks.

### Preliminaries
Download NLTK corpora which are used for defining stopwords and lemmatizing stuff:
```
python -m nltk.downloader stopwords
python -m nltk.downloader reuters
python -m nltk.downloader wordnet
```
Download the english wikipedia dump with:
```
mkdir ./data/wikipedia_dump/
cd ./data/wikipedia_dump/
wget -nc https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```
Preprocess the files by calling the gensim script in the same directory as the dump:
```python
python -m gensim.scripts.make_wiki <PATH_TO_WIKI_DUMP> ./wiki
```
(This will take about 7-8 hours depending on your CPU)
Then decompress the id->word mapping by using bzip2 e.g.:
```
bunzip2 wiki_wordids.txt.bz2
```
Training the LDA will also take several hours. So make sure to use the **--num_workers** flag to speed up the computation accordingly to your computers specs.

### How to fix *ImportError: cannot import name 'zero_gradients' from 'torch.autograd.gradcheck'*:
* Clone https://github.com/BorealisAI/advertorch
* cd into the directory and install the latest Git version with:
* ```python -m pip install -e .```

## Usage:
### 1. Attack
```python
python main.py [-h] [--gpu | -g GPU] [--num_workers | -w WORKERS] [--num_topics | -t TOPICS] [--from_scratch | -s]
```
e.g.
```python
python main.py --attack_id 33 --l2_attack --advs_eps 100
```
### Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -g, --gpu | INT | specifies which GPU should be used [0, 1] (default=0)|
| -b, --batch_size | INT | specifies the batch size (default=128) |
| -e, --epochs | INT | specifies the training epochs (default=100) |
| -l, --learning_rate | FLOAT | specifies the learning rate (default=0.01) |
| -w, --num_workers | INT | number of workers to compute the LDA model (default=4)|
| -t, --num_topics | INT | number of topics which the LDA and the DNN tries to assign the text into |
| -a, --attack_id | INT | specifies the word id for the target of the adversarial attack |
| -ae, --advs_eps | FLOAT | specifies the epsilon for the adversarial attack |
| -ai, --prob_attack | BOOL | flag to activate whole probability distribution target attack |
| -r, --random_test | BOOL | enables random test documents for evaluation |
| -s, --from_scratch | BOOL | flag to ignore every pretrained model and datasets and create everything from scratch |
| -ts, --topic_stacking | BOOL | flat to use the topic stacking attack to evaluate the performance for more than one target topic at the same time |
| -l2, --l2_attack | BOOL | flag to activate the L2 norm attack (rounded floats) instead of the LINF (whole integers) |
| -f, --full_attack | BOOL | flag to attack every topic |
| -v, --verbose | BOOL | flag to set Gensim to verbose mode to print the LDA information during it's training |

### 2. LDA Matching
```python
python lda_matching.py [--benchmark | -b]
```

### 3. LDA Stability Test
```python
python test_lda_stability.py [--corpus_sizes | -cs]
```
e.g.
```python
python test_lda_stability.py --corpus_sizes 0.01 0.1 0.5 1.0
```