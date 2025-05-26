import torch
import faiss
import numpy as np
import json
import os
import argparse
import time
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from itertools import groupby
from operator import itemgetter
from tqdm.auto import tqdm
import psutil
import math
import pdb

import torch.nn.functional as F

# KEEP_DATASETS = ['qmsum', 'legal_case_reports', 'summ_screen_fd', 'qasper_abstract', 'gov_report', 'passage_retrieval']
KEEP_DATASETS = ['qmsum', 'summ_screen_fd', '2wikimqa', 'qasper_abstract', 'qasper_title', 'gov_report', 'passage_retrieval', '2wikimqa']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Retrieval with chunking')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Max sequence length')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--model_name', type=str,
                        default="jinaai/jina-embeddings-v2-small-en",
                        help='Pretrained model')
    parser.add_argument('--num_chunks', type=int, default=5,
                        help='Number of chunks per text')
    parser.add_argument('--chunking_type', type=str, default='default',
                        choices=['late', 'word'],
                        help='Chunking type: default or late')
    parser.add_argument('--max_doc_tokens', type=int, default=None,
                        help='Truncate all documents to this number of words')

    return parser.parse_args()

def load_and_filter_datasets():
    docs = load_dataset("hazyresearch/LoCoV1-Documents", split="test")
    qrys = load_dataset("hazyresearch/LoCoV1-Queries", split="test")
    docs = docs.filter(lambda x: x['dataset'] in KEEP_DATASETS)
    qrys = qrys.filter(lambda x: x['dataset'] in KEEP_DATASETS)
    return docs, qrys

def initialize_model_and_tokenizer(args):
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, config=cfg,
                                      trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    print("Model loaded on device:", dev)
    return model, tok, dev

def log_memory_stats():
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0]/1e9
    else:
        return psutil.virtual_memory().available/1e9


def group_by_dataset(data, keys):
    lst = [{**{k: e[k] for k in keys}, 'idx':i}
           for i,e in enumerate(data)]
    lst.sort(key=itemgetter('dataset'))
    return {k:list(g) for k,g in groupby(lst, key=itemgetter('dataset'))}


def build_indices_word_chunking(model, tok, docs_by_ds, args):
    def encode_and_chunk_by_words(model, tokenizer, texts, args):
        """
        Word-based chunking: split the input text into `args.num_chunks` by words,
        encode each chunk, and normalize the resulting embeddings.
        """
        all_embs = []
        times = []
        mems = []

        for text in texts:
            # Split text into words and divide into chunks
            words = text.split()
            chunk_size = math.ceil(len(words) / args.num_chunks)
            chunks = [
                " ".join(words[i * chunk_size : (i + 1) * chunk_size])
                for i in range(args.num_chunks)
            ]

            chunk_embs = []
            for chunk in chunks:
                # Tokenize and move to device
                inp = tokenizer(
                    [chunk],
                    max_length=args.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inp = {k: v.to(model.device) for k, v in inp.items()}

                # Forward pass
                t0 = time.time()
                with torch.no_grad():
                    out = model(**inp)
                times.append(time.time() - t0)

                # Record memory
                mems.append(log_memory_stats())

                # Mean pooling over tokens
                hidden = out.last_hidden_state  # [1, L, D]
                mask = inp["attention_mask"].unsqueeze(-1)  # [1, L, 1]
                summed = (hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                emb = summed / counts  # [1, D]
                chunk_embs.append(emb)

            # Stack, normalize, and collect
            chunk_tensor = torch.cat(chunk_embs, dim=0)  # [num_chunks, D]
            chunk_tensor = F.normalize(chunk_tensor, p=2, dim=1)
            all_embs.append(chunk_tensor.cpu().half().numpy())

        # Shape: (total_texts * num_chunks, D)
        return np.vstack(all_embs), times, mems

    indices = {}
    id_maps = {}
    fwd_times = []
    mem_stats = {}

    for ds, docs in tqdm(
        docs_by_ds.items(),
        total=len(docs_by_ds),
        desc="Building indices (word chunking)",
    ):
        texts = [d["passage"] for d in docs]
        pids = [d["pid"] for d in docs]

        # Encode and chunk
        embs, times, mem = encode_and_chunk_by_words(model, tok, texts, args)
        fwd_times.extend(times)
        mem_stats[ds] = mem

        # Build FAISS index
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)

        # Map each chunk back to its pid
        id_maps[ds] = np.repeat(pids, args.num_chunks).tolist()
        indices[ds] = idx

    return indices, id_maps, fwd_times, mem_stats


def build_indices_word_chunking_overlap(model, tok, docs_by_ds, args):
    def encode_and_chunk_by_words_overlap(model, tokenizer, texts, args, overlap=100):
        """
        Overlapping word-based chunking: split the input text into `args.num_chunks` by words,
        with each chunk overlapping the previous by `overlap` words.
        """
        all_embs = []
        times = []
        mems = []

        for text in texts:
            words = text.split()
            total_words = len(words)
            chunk_size = math.ceil((total_words + (args.num_chunks - 1) * overlap) / args.num_chunks)
            # Calculate chunk start indices with overlap
            chunk_starts = [i * (chunk_size - overlap) for i in range(args.num_chunks)]
            chunks = []
            for start in chunk_starts:
                end = min(start + chunk_size, total_words)
                chunk = " ".join(words[start:end])
                if chunk:  # avoid empty chunks
                    chunks.append(chunk)

            chunk_embs = []
            for chunk in chunks:
                inp = tokenizer(
                    [chunk],
                    max_length=args.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inp = {k: v.to(model.device) for k, v in inp.items()}

                t0 = time.time()
                with torch.no_grad():
                    out = model(**inp)
                times.append(time.time() - t0)
                mems.append(log_memory_stats())

                hidden = out.last_hidden_state  # [1, L, D]
                mask = inp["attention_mask"].unsqueeze(-1)  # [1, L, 1]
                summed = (hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                emb = summed / counts  # [1, D]
                chunk_embs.append(emb)

            chunk_tensor = torch.cat(chunk_embs, dim=0)  # [num_chunks, D]
            chunk_tensor = F.normalize(chunk_tensor, p=2, dim=1)
            all_embs.append(chunk_tensor.cpu().half().numpy())

        return np.vstack(all_embs), times, mems

    indices = {}
    id_maps = {}
    fwd_times = []
    mem_stats = {}

    for ds, docs in tqdm(
        docs_by_ds.items(),
        total=len(docs_by_ds),
        desc="Building indices (overlapping word chunking)",
    ):
        texts = [d["passage"] for d in docs]
        pids = [d["pid"] for d in docs]

        embs, times, mem = encode_and_chunk_by_words_overlap(model, tok, texts, args, overlap=100)
        fwd_times.extend(times)
        mem_stats[ds] = mem

        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)

        # Map each chunk back to its pid (may be less than args.num_chunks if text is short)
        chunk_counts = [max(1, math.ceil((len(d["passage"].split()) + (args.num_chunks - 1) * 100) / ((len(d["passage"].split()) + (args.num_chunks - 1) * 100) // args.num_chunks))) for d in docs]
        id_map = []
        for pid, count in zip(pids, chunk_counts):
            id_map.extend([pid] * count)
        id_maps[ds] = id_map[:embs.shape[0]]
        indices[ds] = idx

    return indices, id_maps, fwd_times, mem_stats

def build_indices_late_chunking(model, tok, docs_by_ds, args):
    def encode_and_chunk_chunking(model, tokenizer, texts, args):
        """
        Late chunking: for each text, encode full sequence, take CLS vector,
        then split its feature dimension into `args.num_chunks` slices.
        """
        all_embs = []
        times = []
        mems = []
    
        for text in texts:
            # tokenize & move to device
            inp = tokenizer(
                [text],
                max_length=args.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inp = {k: v.to(model.device) for k, v in inp.items()}


            # forward
            t0 = time.time()
            with torch.no_grad():
                out = model(**inp)
            times.append(time.time() - t0)

            # record memory
            mems.append(log_memory_stats())

            # take CLS embedding
            cls_vec = out.last_hidden_state # [1, L, D]
            seq_len = cls_vec.size(1)
        
            # split into num_chunks and mean pool each chunk
            chunk_embs = []
            for i in range(args.num_chunks):
                start = (i * seq_len) // args.num_chunks
                end = ((i + 1) * seq_len) // args.num_chunks
                chunk = cls_vec[:, start:end, :]  # [1, chunk_size, dim]

                chunk_mean = chunk.mean(dim=1)  # [1, 1, dim]
                chunk_embs.append(chunk_mean)  # [1, dim]
            
            # stack & flatten to [num_chunks, sub_dim]
            chunks = torch.cat(chunk_embs, dim=0)
            # normalize each chunk
            chunks = F.normalize(chunks, p=2, dim=1)

            all_embs.append(chunks.cpu().half().numpy())

        # shape: (total_texts * num_chunks, sub_dim)
        return np.vstack(all_embs), times, mems

    indices = {}
    id_maps = {}
    fwd_times = []
    mem_stats = {}

    for ds, docs in tqdm(
        docs_by_ds.items(),
        total=len(docs_by_ds),
        desc="Building indices (late chunking)"
    ):
        texts = [d["passage"] for d in docs]
        pids = [d["pid"] for d in docs]

        # encode + chunk
        embs, times, mem = encode_and_chunk_chunking(model, tok, texts, args)
        fwd_times.extend(times)
        mem_stats[ds] = mem

        # build FAISS
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)

        # map each chunk back to its pid
        id_maps[ds] = np.repeat(pids, args.num_chunks).tolist()
        indices[ds] = idx

    return indices, id_maps, fwd_times, mem_stats


def encode_query(model, tokenizer, text, args):
    """Encode a query without chunking - just use mean pooling over the entire text"""
    inp = tokenizer([text], max_length=args.max_length,
                    padding=True, truncation=True, return_tensors='pt')
    inp = {k:v.to(model.device) for k,v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    mask = inp['attention_mask']
    # Mean pooling across all tokens
    embedding = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().half().numpy()

def evaluate_retrieval(model, tok, docs_by_ds, qrys_by_ds,
                       indices, id_maps, args):
    results = {}
    for ds, qrys in qrys_by_ds.items():
        if ds not in indices:
            continue
        idx = indices[ds]
        pid_map = id_maps[ds]
        hits1, hits3 = 0, 0
        for q in tqdm(qrys):
            qry_embs = encode_query(model, tok, q['query'], args)
            k = args.num_chunks * 10  
            D, I = idx.search(qry_embs, k)
            cand = {}
            for ci, scores in zip(I, D):
                for j, sc in zip(ci, scores):
                    pid = pid_map[j]
                    cand[pid] = max(cand.get(pid, -1e9), sc)

                    if len(cand) >= 3: # retrieve top3 
                        break 

            sorted_p = sorted(cand.items(), key=lambda x:-x[1])
            top = [p for p,_ in sorted_p[:3]]
            if q['answer_pids'][0] in top[:1]:
                hits1 += 1
            if any(a in top for a in q['answer_pids']):
                hits3 += 1
        n = len(qrys)
        results[ds] = {
            'hit@1': round(hits1/n, 3),
            'hit@3': round(hits3/n, 3),
            'total': n
        }
    # aggregate
    all1 = round(np.mean([v['hit@1'] for v in results.values()]), 3)
    all3 = round(np.mean([v['hit@3'] for v in results.values()]), 3)
    metrics = {'overall': {'hit@1': all1, 'hit@3': all3},
               'by_dataset': results}
    return metrics

def save_metrics(metrics, args):
    os.makedirs(args.out_dir, exist_ok=True)

    short_model_name = args.model_name.split('/')[-1]

    # Create metrics filename
    base_filename = f"{short_model_name}_{args.max_length}_{args.num_chunks}_{args.chunking_type}_{args.max_doc_tokens}"
    path = os.path.join(args.out_dir, f"{base_filename}.json")

    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {path}")
    print(metrics)

def truncate_docs_by_tokens(docs_by_ds, tokenizer, max_doc_tokens):
    """
    Truncate documents in docs_by_ds at the token level if max_doc_tokens is specified.
    """
    if max_doc_tokens is not None:
        for ds, docs in docs_by_ds.items():
            for doc in docs:
                tokens = tokenizer(
                    doc['passage'],
                    max_length=max_doc_tokens,
                    truncation=True,
                    return_tensors=None,
                    add_special_tokens=True
                )
                doc['passage'] = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

def truncate_docs_by_words(docs_by_ds, tokenizer=None, max_doc_tokens=None):
    """
    Truncate documents in docs_by_ds at the word level if max_doc_words is specified.
    """
    if max_doc_tokens is not None:
        for ds, docs in docs_by_ds.items():
            for doc in docs:
                words = doc['passage'].split()
                if len(words) > max_doc_tokens:
                    doc['passage'] = " ".join(words[:max_doc_tokens])


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.num_chunks = 30
            self.max_length = 4092*2
            self.out_dir = "results2"
            # self.model_name = "Alibaba-NLP/gte-base-en-v1.5"
            self.model_name = "jinaai/jina-embeddings-v2-small-en"
            self.chunking_type = "late"
            # self.max_doc_length = None
            self.max_doc_tokens = self.max_length
    

    # args = Args() # for notebooks
    args = parse_arguments()
    
    docs, qrys = load_and_filter_datasets()
    print("Filtered datasets", len(docs), len(qrys))
    model, tokenizer, dev = initialize_model_and_tokenizer(args)

    docs_by_ds = group_by_dataset(docs, ['dataset','pid','passage'])
    qrys_by_ds = group_by_dataset(qrys, ['dataset','qid','query','answer_pids'])

    truncate_docs_by_words(docs_by_ds, tokenizer, args.max_doc_tokens)

    if args.chunking_type == 'late':
        indices, id_maps, fwd_times, mem_stats = build_indices_late_chunking(
        model, tokenizer, docs_by_ds, args)

    elif args.chunking_type == 'word':
        indices, id_maps, fwd_times, mem_stats = build_indices_word_chunking(
            model, tokenizer, docs_by_ds, args)
    else:
        raise ValueError("Invalid chunking type. Use 'late' or 'default'.")

    print("Chunking method used", args.chunking_type)

    metrics = evaluate_retrieval(
        model, tokenizer, docs_by_ds, qrys_by_ds, indices, id_maps, args)
    save_metrics(metrics, args)


