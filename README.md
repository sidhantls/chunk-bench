# chunk-bench

Evaluating different strategies for late chunking on long-context documents. Methods tested:

* Word Chunking: Input word based chunking
* Late Chunking: Output vector based chunking

## Method: 
* **Word Chunking**: 
    - The document is divided into `n_chunks` number of subdocuments.
    - Vectors for these subdocuments are generated individually and indexed.
    - Search is performed over `n_chunks` number of vectors, and argmax is used to select document 

* **Late Chunking**: 
    - The document is fully encoded into all token vectors.
    - These token vectors are separated into `n_chunks` number of chunks. Each of almost equal length. 
    - Mean pooling is performed for each chunk separately to get `n_chunks` chunked vectors. 

### Metrics: 
* Topk @ 1, 3

* Late Chunking: A 
## Requirements
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage
### Parameters
- `max_length`: Maximum tokens the model can process.
- `max_doc_tokens`: Truncates all datasets to be less than or equal to this number of words.

### Examples
Test late chunking that outputs 5 chunk vectors:
```bash
python benchmark_chunking.py --model_name="jinaai/jina-embeddings-v2-small-en" --num_chunks=5 --max_length=4092 --chunking_type="late" --max_doc_tokens=4092
```

Test word chunking, splits document into 5 chunks at the word level, outputs 5 chunk vectors:
```bash
python benchmark_chunking.py --model_name="jinaai/jina-embeddings-v2-small-en" --num_chunks=5 --max_length=4092 --chunking_type="word" --max_doc_tokens=4092
```

Baseline, no chunking:
```bash
python benchmark_chunking.py --model_name="jinaai/jina-embeddings-v2-small-en" --num_chunks=1 --max_length=4092 --chunking_type="word" --max_doc_tokens=4092
```