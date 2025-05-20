#!/bin/bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
-m FlagEmbedding.reranker.run \
--output_dir ./checkpoints/xxx \
--model_name_or_path BAAI/bge-reranker-base \
--train_data ./dataset/nq/xxx/FinetuneDataset_for_flagembedding.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 2 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 7 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 100 \
--save_strategy no \
--save_steps 1000

# explanations of arguments
# train_group_size: number of positive and negative samples
# resume_from_checkpoint: path to the checkpoint to resume from
# save_strategy: no, steps, or epoch