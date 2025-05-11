# chunk-bench

Evaluating different strategies for late chunking on long-context documents. Methods tested:

* Word Chunking: Input word based chunking
* Late Chunking: Output vector based chunking. Based on, Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

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
* Dataset: [Loco-v1](https://huggingface.co/datasets/hazyresearch/LoCoV1-Documents)

### Implementation Caveats: 
Description of how the implemented late chunking deviates from the [paper](https://arxiv.org/abs/2409.04701) 

Late Chunking: 
* Chunk documents based on the desired number of chunks rather than the token length of each chunk:
    * In practical industrial applications, documents are often divided into a small number of chunks to optimize efficiency. This approach greatly reduces memory storage requirements and minimizes retrieval latency.
    * The original paper uses token-based chunking, so the metrics in this implementation may not align perfectly with the results reported in their table.

## Requirements
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage
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

### Parameters
- **`max_length`**: *(int, default=4096)*  
    Maximum number of tokens the model can process in a single chunk.

- **`out_dir`**: *(str, default="results")*  
    Output directory where results will be saved.

- **`model_name`**: *(str, default="jinaai/jina-embeddings-v2-small-en")*  
    Name or path of the pretrained embedding model.

- **`num_chunks`**: *(int, default=5)*  
    Number of chunks each document will be split into.

- **`chunking_type`**: *(str, default="default", choices=["late", "word"])*  
    Chunking strategy:
    - **`word`**: Splits the document into word-based chunks before embedding.
    - **`late`**: Embeds the entire document, then splits the embedding into chunks.

- **`max_doc_tokens`**: *(int, default=None)*  
    If set, truncate each document to this number of words before chunking or embedding.
