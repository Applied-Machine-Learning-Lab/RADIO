import argparse
import os
import json
import random
import numpy as np
import torch
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator
from transformers import AutoTokenizer
from utils.classes import GenerateDatasetPipeline
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from vllm import LLM, SamplingParams
from nltk.tokenize import PunktSentenceTokenizer
import re

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

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
        # sampling index is [3,4,6,9,13,18]
        entry_list = [{
            "query": question,
            "pos": [truncated_sorted_docs[0][0]],
            "neg": [truncated_sorted_docs[3][0], truncated_sorted_docs[4][0], truncated_sorted_docs[6][0], truncated_sorted_docs[9][0], truncated_sorted_docs[13][0], truncated_sorted_docs[18][0]]
        }]
        return entry_list


def get_docs(args):
    # get the top n samples, and store the question, retrieval result, and golden answers into a file
    # *************the generation process is ignored****************
    all_split = get_dataset(args.config)
    test_data = all_split["train"]
    test_data.data = test_data.data[:20000] # top 20000 samples
    prompt_templete = PromptTemplate(
        args.config,
        system_prompt = "None",
        user_prompt="Question: {question}\nContextual Passages: {reference}\nWhy the answer is {answer}?\Reason:",
    )
    generator = get_generator(args.config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = "<lendoftext|>"
    tokenizer.pad_token_id = 128009
    tokenizer.padding_side = "left"
    generator.tokenizer = tokenizer

    pipeline = GenerateDatasetPipeline(args.config, prompt_template=prompt_templete, generator=generator)

    output_dataset = pipeline.run(test_data, do_eval=True)
    output_dataset.save("./dataset/{}/train_docs_20000.json".format(args.dataset))

def generate_dataset(args):
    if args.method == 'reason':
        path = "./dataset/{}/train_docs_20000.json".format(args.dataset)
        output_path = "./dataset/{}/reason/FinetuneDataset_for_flagembedding.jsonl".format(args.dataset)
        cache_output_path = "./dataset/{}/reason/score_cache.jsonl".format(args.dataset)
        with open(path, 'r', encoding='utf-8') as file:
            retrieval_data = json.load(file)
        path = "./dataset/{}/reason/train_with_reason.jsonl".format(args.dataset)
        with open(path, 'r', encoding='utf-8') as file:
            generation_data = file.readlines()

        # Load the embedding model
        embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=args.device)
        output_list = []
        cache_list = []
        for i in tqdm(range(len(retrieval_data)), desc="Generating dataset"):
            question = retrieval_data[i]["question"]
            generation_entry = json.loads(generation_data[i])
            reason = str(generation_entry["question"]) + '. ' + str(generation_entry["golden_answers"]) + '. ' + str(generation_entry["reason"])
            docs = [retrieval_data[i]['output']['retrieval_result'][j]['contents'] for j in range(len(retrieval_data[i]['output']['retrieval_result']))]

            # Encode the reason and docs using the embedding model
            reason_embedding = embedding_model.encode([reason])['dense_vecs']
            docs_embeddings = embedding_model.encode([str(doc) for doc in docs])['dense_vecs']

            # Calculate similarity between reason and docs
            similarity = reason_embedding @ docs_embeddings.T
            similarity_scores = similarity.flatten().tolist()

            # Sort the docs based on similarity score from high to low
            sorted_docs = sorted(zip(docs, similarity_scores), key=lambda x: x[1], reverse=True)

            rank_idx = np.argsort(similarity_scores)[::-1]

            training_entry = sample(question, sorted_docs, 'shift', batch_size=5, truncate_length=20)
            # Append the training entry to output list
            for entry in training_entry:
                output_list.append(entry)

            cache_entry = {
                "query": question,
                "content": sorted_docs
            }

            cache_list.append(cache_entry)

        # Save the output list as a jsonl file in the required format
        save_jsonl(output_list, output_path)
        save_jsonl(cache_list, cache_output_path)

    elif args.method == "retrieval":
        path = "./dataset/{}/train_docs_20000.json".format(args.dataset)
        output_path = "./dataset/{}/retrieval/FinetuneDataset_for_flagembedding.jsonl".format(args.dataset)
        cache_output_path = "./dataset/{}/retrieval/score_cache.jsonl".format(args.dataset)
        with open(path, 'r', encoding='utf-8') as file:
            retrieval_data = json.load(file)

        output_list = []
        cache_list = []
        for i in tqdm(range(len(retrieval_data))):
            question = retrieval_data[i]["question"]
            correct_answer = retrieval_data[i]["golden_answers"]
            correct_answer = ','.join(correct_answer)
            docs = [retrieval_data[i]['output']['retrieval_result'][j]['contents'] for j in range(len(retrieval_data[i]['output']['retrieval_result']))]
            retrieval_scores = retrieval_data[i]['output']['retrieval_score']

            sorted_docs = list(zip(docs, retrieval_scores))
            training_entry = sample(question, sorted_docs, 'shift', batch_size=5, truncate_length=20)

            # # Append the training entry to output list
            for entry in training_entry:
                output_list.append(entry)
            # output_list.append(training_entry)

            cache_entry = {
                "query": question,
                "content": sorted_docs
            }
            cache_list.append(cache_entry)

        # Save the final output list as a jsonl file in the required format
        save_jsonl(output_list, output_path)
        save_jsonl(cache_list, cache_output_path)
        print(f"Generated dataset saved to {output_path}")
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='xxx/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f')
    parser.add_argument("--retriever_path", type=str, default='xxx/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/1c644c92ad3ba1efdad3f1451a637716616a20e8')
    parser.add_argument("--dataset", type=str, default='nq', help='nq, triviaqa, mmlu')
    parser.add_argument("--phase", type=str, help="get_docs or generate_dataset", default='generate_dataset')
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--method", type=str, default='reason')
    args = parser.parse_args()

    if args.phase == "get_docs":
        config_dict = {
            "data_dir": "dataset/",
            "index_path": "indexes/e5_Flat.index",
            "corpus_path": "indexes/retrieval-corpus/wiki-18.jsonl",
            "model2path": {
                "e5": args.retriever_path, 
                "llama3-8B-instruct": args.model_path
                },
            "generator_model": "llama3-8B-instruct",
            "retrieval_method": "e5",
            "metrics": ["em", "f1"],
            "retrieval_topk": 50,
            "save_intermediate_data": True,
            "dataset_name": args.dataset,
            # "test_sample_num": 10,
            "gpu_id": 0,
            "generation_params": {
                "max_tokens": 512
            },
            "faiss_gpu": False,
            "retrieval_batch_size": 1024,
            "split": ["train", "dev", "test"],
        }

        args.config = Config(config_dict=config_dict)
        get_docs(args)
    elif args.phase == "generate_dataset":
        generate_dataset(args)