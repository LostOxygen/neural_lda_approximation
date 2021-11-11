# Neural LDA Approximation
Approximation of the latent dirichlet allocation per deep neural networks.

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
