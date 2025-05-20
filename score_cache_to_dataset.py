import pandas as pd
import json

dataset_save_path = "xxx/dataset/nq/reason+retrieval/FinetuneDataset_for_flagembedding_reason9.jsonl"
cache_output_path = '../dataset/triviaqa/reason/score_cache.jsonl'
cache_output_retrieval = '../dataset/triviaqa/retrieval/score_cache.jsonl'

def sample(question, sorted_docs, method, batch_size=5, truncate_length=20):
    if method == 'iteration':
        # sorted_docs: (doc, score)
        truncated_sorted_docs = sorted_docs[:truncate_length]
        batch_num = len(truncated_sorted_docs) // batch_size
        entry_list = []
        for group in range(batch_num):
            start = group * batch_size
            for i in range(start, start + batch_size - 1):
                sampled_negative_docs = []
                for j in range(i+1, start + batch_size):
                    sampled_negative_docs.append(truncated_sorted_docs[j][0])
                entry = {
                    "query": question,
                    "pos": [truncated_sorted_docs[i][0]],
                    "neg": sampled_negative_docs
                }
                entry_list.append(entry)
        return entry_list
    elif method == 'shift':
        truncated_sorted_docs = sorted_docs[:truncate_length]
        entry_list = [{
            "query": question,
            "pos": [truncated_sorted_docs[0][0]],
            "neg": [truncated_sorted_docs[3][0], truncated_sorted_docs[4][0], truncated_sorted_docs[6][0], truncated_sorted_docs[9][0], truncated_sorted_docs[13][0], truncated_sorted_docs[18][0]]
        }]
        return entry_list
    elif method == 'shift_all_neg':
        truncated_sorted_docs = sorted_docs[:truncate_length]
        entry_list = [{
            "query": question,
            "pos": [truncated_sorted_docs[0][0]],
            "neg": [truncated_sorted_docs[i][0] for i in range(3, 20)]
        }]
        return entry_list


cache_list_reason = []
with open(cache_output_path, 'r', encoding='utf-8') as file:
    for line in file:
        cache_list_reason.append(json.loads(line))

cache_list_retrieval = []
with open(cache_output_retrieval, 'r', encoding='utf-8') as file:
    for line in file:
        cache_list_retrieval.append(json.loads(line))

output_list = []
import numpy as np
for i in range(len(cache_list_reason)):
    sorted_docs_reason = cache_list_reason[i]['content']
    sorted_docs_retrieval = cache_list_retrieval[i]['content']
    question = cache_list_reason[i]['query']

    scores_reason = np.array([doc[1] for doc in sorted_docs_reason])
    scores_reason = (scores_reason - scores_reason.min()) / (scores_reason.max() - scores_reason.min())
    scores_retrieval = np.array([doc[1] for doc in sorted_docs_retrieval])
    scores_retrieval = (scores_retrieval - scores_retrieval.min()) / (scores_retrieval.max() - scores_retrieval.min())

    sorted_docs = []
    for j in range(len(sorted_docs_reason)):
        score_reason = scores_reason[j]
        doc_reason = sorted_docs_reason[j][0]
        for k in range(len(sorted_docs_retrieval)):
            score_retrieval = scores_retrieval[k]
            doc_retrieval = sorted_docs_retrieval[k][0]
            if doc_reason == doc_retrieval:
                score = 0.5 * score_reason + 0.5 * score_retrieval
                sorted_docs.append((doc_reason, score))
                break
    
    sorted_docs = sorted(sorted_docs, key=lambda x: x[1], reverse=True)

    entry = sample(question, sorted_docs, 'shift', 5, 20)
    for e in entry:
        output_list.append(e)

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')
save_jsonl(output_list, dataset_save_path)