# Neural LDA Approximation
Approximation of the latent dirichlet allocation per deep neural networks.

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
Then compress the .mm ouput files and decompress the id->word mapping by using bzip2 e.g.:
```
bzip2 wiki_bow.mm
bunzip2 wiki_wordids.txt.bz2
```
And delete every file except those two.
Training the LDA will also take several hours. So make sure to use the **--num_workers** flag to speed up the computation accordingly to your computers specs.

### Usage
```python
python main.py [-h] [--gpu | -g GPU] [--num_workers | -w WORKERS] [--num_topics | -t TOPICS] [--from_scratch | -s]
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
| -f, --freq_id | INT | specifies the word id of which the frequency gets changed to 1000 (default=None) |
| -q, --freq| INT | specifies the new frequency of the word with the chosen freq_id |
| -r, --random_test | enables random test documents for evaluation |
| -s, --from_scratch | BOOL | flag to ignore every pretrained model and datasets and create everything from scratch |
| -v, --verbose | BOOL | flag to set Gensim to verbose mode to print the LDA information during it's training |
