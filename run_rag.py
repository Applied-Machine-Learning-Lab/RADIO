import argparse
import os
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator
from transformers import AutoTokenizer
from utils.classes import CustomSequentialPipeline, mmluPipeline

def main(args):
    if args.dataset == 'mmlu':
        print('dataset is mmlu')
        all_split = get_dataset(args.config)
        test_data = all_split["test"]
        prompt_templete = PromptTemplate(
            args.config,
            system_prompt="Answer the question based on the given document. \
                            Only give me the option (A/B/C/D) and do not output any other words. \
                            \nThe following are given documents.\n\n{reference}",
            user_prompt="Question: {question}\nAnswer:",
        )
        generator = get_generator(args.config)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = "<lendoftext|>"
        tokenizer.pad_token_id = 128009
        tokenizer.padding_side = "left"
        generator.tokenizer = tokenizer
        pipeline = mmluPipeline(args.config, prompt_template=prompt_templete, generator=generator)
        output_dataset = pipeline.run(test_data, do_eval=True, pred_process_fun=None)
    else:
        all_split = get_dataset(args.config)
        test_data = all_split["test"]
        prompt_templete = PromptTemplate(
            args.config,
            system_prompt="Answer the question based on the given document. \
                            Only give me the answer and do not output any other words. \
                            \nThe following are given documents.\n\n{reference}",
            user_prompt="Question: {question}\nAnswer:",
        )
        generator = get_generator(args.config)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = "<lendoftext|>"
        tokenizer.pad_token_id = 128009
        tokenizer.padding_side = "left"
        generator.tokenizer = tokenizer

        pipeline = SequentialPipeline(args.config, prompt_template=prompt_templete, generator=generator)

        output_dataset = pipeline.run(test_data, do_eval=True)
        # output_dataset.save(os.path.join(args.config['save_dir'], 'intermediate_data.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='xxx/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f')
    parser.add_argument("--retriever_path", type=str, default='xxx/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/1c644c92ad3ba1efdad3f1451a637716616a20e8')
    parser.add_argument("--rerank_model_name", type=str, default="bge-reranker-base")
    parser.add_argument("--rerank_model_path", type=str, default='xxx/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70/')
    parser.add_argument("--gpu_id", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--framework", type=str, default="fschat")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    config_dict = {
        "data_dir": "dataset/",
        "index_path": "indexes/e5_Flat.index",
        "corpus_path": "indexes/retrieval-corpus/wiki-18.jsonl",
        "model2path": {
            # "e5": args.retriever_path, 
            # "llama3-8B-instruct": args.model_path
            },
        # "generator_model": "llama3-8B-instruct",
        "framework": args.framework,
        "generator_model": args.model_path if args.framework == "fschat" else args.openai_model,
        "openai_setting": {
            "api_key": "sk-xxx",
            "base_url": "https://xxx.xxx.xxx"
        },
        # "retrieval_method": "e5",
        "retrieval_method": args.retriever_path,
        "metrics": ["em", "f1"],
        "retrieval_topk": 20,
        "save_intermediate_data": True,
        "dataset_name": args.dataset,
        # "test_sample_num": 100,
        "gpu_id": args.gpu_id,
        "generation_params": {
            "max_tokens": args.max_tokens,
        },
        "use_reranker": True,
        "rerank_model_name": "bge-reranker-base",
        "rerank_model_path": args.rerank_model_path,
        "rerank_topk": 5,
        "rerank_batch_size": 128,
        "save_retrieval_cache": False,
    }

    args.config = Config(config_dict=config_dict)
    main(args)