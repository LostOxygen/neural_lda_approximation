# Neural LDA Approximation
Approximation of the latent dirichlet allocation per deep neural networks.

### Preliminaries
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
| -g, --gpu | INT | specifies which GPU should be used [0, 1] |
| -w, --num_workers | INT | number of workers to compute the LDA model (default=4)|
| -t, --num_topics | INT | number of topics which the LDA and the DNN tries to assign the text into |
| -s, --from_scratch | BOOL | flag to ignore every pretrained model and datasets and create everything from scratch |
