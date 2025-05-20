# RADIO

## Environment

```bash
# install from github
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .[core] 

pip install vllm==0.6.0

pip install FlagEmbedding==1.2.11

pip install evaluate

# Install all extra dependencies
pip install flashrag-dev[full]

# install faiss
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install -U bitsandbytes

pip install pandarallel
```

## Create Index

Please first download the corpus wiki18 from [Huggingface](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus) or [ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files), then run the following script.

```bash
CUDA_VISIBLE_DEVICES=2 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path xxx/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/1c644c92ad3ba1efdad3f1451a637716616a20e8/ \
    --corpus_path indexes/retrieval-corpus/wiki-18.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat
```

## File Structure

```
--utils
    |--basic_config.yaml. The file containing all parameters in flashrag
    |--classes.py. Customized RAG pipelines.
--dataset_generation.py. Python script to generate fine-tuning datasets.
--llm_generation.py. Python script to generate rationales for datasets.
--mmlu_evaluation.py. Python script to evaluate mmlu dataset on different categories.
--mmlu_preprocess.py. Python script to preprocess mmlu dataset.
--run_finetune.sh. Script to fine-tune rerankers.
--run_rag.py. Python script to run rag pipelines.
--score_cache_to_dataset.py. Python script to integrate different scores to a dataset.
```

## Guildlines

### Generate Rationales

`python llm_generation.py --api_key=xxx --base_url=xxx --dataset=nq --model=gpt-4o-mini`

### Generate Dataset

1. Generate score files on rationales and retrieval separately

    `python dataset_generation.py --dataset=xxx --phase=get_docs`

    `python dataset_generation.py --dataset=xxx --phase=generate_dataset --method=reason`

    `python dataset_generation.py --dataset=xxx --phase=generate_dataset --method=retrieval`

2. Integrate rationale-based and retrieval scores and generate dataset

    `python score_cache_to_dataset.py`

### Fine-tune Rerankers

`sh run_finetune.sh`

### Run RAG Pipelines

`python run_rag.py --rerank_model_path=xxx --dataset=nq`

If you want to use generators with openai api, you can try 

`python run_rag.py --rerank_model_path=xxx --framework=openai --openai_model=gpt-4o-mini`
