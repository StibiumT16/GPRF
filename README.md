# GPRF
Generalized Pseudo-Relevance Feedback

## 0. Requirements
We implement the training and RAG pipeline based on [Swift](https://github.com/modelscope/ms-swift) and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) respectively. Please install them according to their requirements.

## 1. Data Preparation
We primarily use the [BEIR](https://github.com/beir-cellar/beir) and [DL2019/DL2020](https://microsoft.github.io/msmarco/) datasets, which can be downloaded from their official websites and placed in the `./BEIR` directory. We have made some adjustments to the data format, and the file structure is organized as follows:
```
BEIR/
├── {dataset name}/
│   ├── corpus.jsonl
|   ├── queries.jsonl
|   └── qrels/
|       ├── (train.tsv)
|       ├── (dev.tsv)
│       └── test.tsv
```

The lines in `corpus.jsonl` are as follows:
```json
{"id": "0", "contents": "The presence of communication..."}
```

The lines in `queries.jsonl` are as follows:
```json
{"_id": "1108939", "text": "what slows down the flow of blood"}
```

The format of qrels files are as follows:
```
query-id	corpus-id	score
19335	1017759	0
19335	1082489	0
19335	109063	0
19335	1720389	1
......
```

We provide an example in `./BEIR/msmarco`. Note that the corpora of DL2019 and DL2020 are the same as MSMARCO in BEIR. You can set the path manually or just create a link.

## 2. How to retrieve?
### 2.1. Build the Index
First, we need to build indexes for different retrievers. Below we give the commands for building indexes for BM25, E5, and BGE respectively:

```bash
mkdir -p data/msmarco/index

python src/index.py \
    --retrieval_method bm25 \
    --model_path bm25 \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --save_dir data/msmarco/index/ \

python src/index.py \
    --retrieval_method e5-base-v2 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --save_dir data/msmarco/index/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat
    
python src/index.py \
    --retrieval_method bge-base-en-v1.5 \
    --model_path BAAI/bge-base-en-v1.5 \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --save_dir data/msmarco/index/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method cls \
    --faiss_type Flat
```

### 2.2. Retrieve
```bash
mkdir -p data/msmarco/original

python src/retrieve.py \
    --qrels_path BEIR/msmarco/qrels/test.tsv \
    --queries_path BEIR/msmarco/queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --index_path data/msmarco/index/bm25 \
    --output_path data/msmarco/original/test_bm25.jsonl \
    --topk 100 \
    --retriever bm25 \
    --retrieval_method bm25 \
    --split test

python src/retrieve.py \
    --qrels_path BEIR/msmarco/qrels/test.tsv \
    --queries_path BEIR/msmarco/queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --index_path data/msmarco/index/e5-base-v2_Flat.index \
    --output_path data/msmarco/original/test_e5-base-v2.jsonl \
    --topk 100 \
    --retriever e5-base-v2 \
    --retrieval_method intfloat/e5-base-v2 \
    --pooling_method mean \
    --split test

python src/retrieve.py \
    --qrels_path BEIR/msmarco/qrels/test.tsv \
    --queries_path BEIR/msmarco/queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --index_path data/msmarco/index/e5-base-v2_Flat.index \
    --output_path data/msmarco/original/test_bge-base-en-v1.5.jsonl \
    --topk $topk \
    --retriever bge-base-en-v1.5 \
    --retrieval_method BAAI/bge-base-en-v1.5 \
    --pooling_method cls \
    --split test
```

### 2.3. Evaluation
Take BM25 as an example:
```bash
python src/eval.py \
    --qrels_path BEIR/msmarco/qrels/test.tsv \
    --result_path data/msmarco/original/test_bm25.jsonl
```


## 3. Our Training Pipeline
### 3.1. Rejection Sampling
First conduct first-stage retrieval on MSMARCO Training Set following `2.2.` (dataset=msmarco, split=train), results are saved as: `./data/msmarco/original/train_${retriever}.jsonl`.

When we exploit an LLM to sample reformulations:
```bash
retriever=bm25 # e5-base-v2

CUDA_VISIBLE_DEVICES=0,1 python src/llm.py \
    --method qrg \
    --dataset msmarco \
    --qrels_path BEIR/msmarco/qrels/train.tsv \
    --queries_path BEIR/msmarco/queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --result_path data/msmarco/original/train_${retriever}.jsonl \
    --output_path data/train/${retriever}_queries.jsonl \
    --result_topk 10 \
    --gpu_num $gpu_num \
    --generator_model <your Generator LLM> \ 
    --generator_model_path <your Generator LLM path> \
    --generator_max_input_len 8192 \
    --max_tokens 512 \
    --do_sample \
    --n 10 \
    --sample_size 200000
```

After generation, we directly record the rewritten queries that achieve the best performance in downstream retrieval tasks based on the retrieval results:
```bash
python src/rej.py \
    --qrels_path BEIR/msmarco/qrels/train.tsv \
    --queries_path data/train/bm25_queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --index_path data/msmarco/index/bm25 \
    --output_path data/train/bm25_rej.jsonl \
    --retriever bm25 \
    --retrieval_method bm25 \
    --pooling_method mean

python src/rej.py \
    --qrels_path BEIR/msmarco/qrels/train.tsv \
    --queries_path data/train/e5-base-v2_queries.jsonl \
    --corpus_path BEIR/msmarco/corpus.jsonl \
    --index_path data/msmarco/index/e5-base-v2 \
    --output_path data/train/e5-base-v2_rej.jsonl \
    --retriever e5-base-v2 \
    --retrieval_method intfloat/e5-base-v2 \
    --pooling_method mean
```

We provide examples of a generated result and a filtered result in `./data/train/bm25_queries.json` and `./data/train/bm25_rej.json`, respectively. In `./data/train/bm25_queries.json`, the field `num` with a value of 0 denotes the original query, while values from 1 to 10 represent ten different sampled reformulations.

After filtering the data, we can construct the data for SFT and RL:
```bash
cd train
python sft_data.py
python rl_data.py
```


### 3.2. Supervised Fine-tuning
```bash
gpu_ids=1,2
gpu_num=2

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
swift sft \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train_type full \
    --dataset 'train_data/gprf_sft' \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_steps 300 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 3072 \
    --output_dir output/GPRF_SFT/Llama3B \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2_offload \
    --save_only_model true \
    --use_liger_kernel true
```

### 3.3. Reinforcement Learning
In RL training, first launch two retrievers separately:
```bash
# Use the RAG environment!
python src/retrieval_server.py \
    --index_path ../data/msmarco/index/e5-base-v2_Flat.index \
    --corpus_path ../BEIR/msmarco/corpus.jsonl \
    --topk 10 \
    --retrieval_method intfloat/e5-base-v2 \
    --port 11451
```

```bash
# Use the RAG environment!
python src/retrieval_server.py \
    --index_path ../data/msmarco/index/bm25 \
    --corpus_path ../BEIR/msmarco/corpus.jsonl \
    --topk 10 \
    --retrieval_method bm25 \
    --port 11452
```

Then you need to start the rollout step：
```bash
CUDA_VISIBLE_DEVICES=0 swift rollout \
    --model <your SFT checkpoint path> \
    --port 20355
```

Finally, you can start training the model:
```bash
gpu_ids=1,2,3,4
gpu_num=4

CUDA_VISIBLE_DEVICES=$gpu_ids \
NPROC_PER_NODE=$gpu_num \
swift rlhf \
    --rlhf_type grpo \
    --model <your SFT checkpoint path> \
    --external_plugins plugin.py \
    --reward_funcs external_ndcg  \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 20355 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'train_data/gprf_rl' \
    --max_length 4096 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --save_steps 500 \
    --save_only_model true \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/GPRF/Llama3B \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --deepspeed zero3_offload \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1
```

## 4. Inference & Evaluation
First, you need to perform query rewriting based on the trained model using the original queries and the first-stage retrieval results:
```bash
mkdir -p data/dl2019/rewrite

CUDA_VISIBLE_DEVICES=0 python src/llm.py \
    --method gprf \
    --dataset dl2019 \
    --qrels_path BEIR/dl2019/qrels/test.tsv \
    --queries_path BEIR/dl2019/queries.jsonl \
    --corpus_path BEIR/dl2019/format_corpus.jsonl \
    --result_path data/dl2019/original/test_bm25.jsonl \
    --output_path data/dl2019/rewrite/gprf_bm25_Llama3B.jsonl \
    --result_topk 10 \
    --gpu_num 1 \
    --generator_model Llama-3.2-3B-Instruct \
    --generator_model_path "<your GPRF model path>" \
    --generator_max_input_len 8192 \
    --max_tokens 512 \
    --do_sample \
    --n 10
```

The generated rewriting result format is as follows:
```json
{"id": 47923, "feedback": ["Synaptic knobs are the rounded areas...", "Synaptic knobs are the terminal part...", ......]}
```

Then, we can integrate them into our retrieval process and evaluate the results:
```bash
python src/retrieve.py \
    --qrels_path BEIR/dl2019/qrels/test.tsv \
    --queries_path BEIR/dl2019/queries.jsonl \
    --corpus_path BEIR/dl2019/format_corpus.jsonl \
    --index_path data/dl2019/index/bm25 \
    --output_path data/dl2019/result/GPRF_bm25.jsonl \
    --feedback_path data/dl2019/rewrite/gprf_bm25_Llama3B.jsonl \
    --feedback_topk 10 \
    --topk 100 \
    --retriever bm25 \
    --retrieval_method bm25 \
    --split test \
    --feedback_is_query

python src/retrieve.py \
    --qrels_path BEIR/dl2019/qrels/test.tsv \
    --queries_path BEIR/dl2019/queries.jsonl \
    --corpus_path BEIR/dl2019/format_corpus.jsonl \
    --index_path data/dl2019/index/e5-base-v2_Flat.index \
    --output_path data/dl2019/result/GPRF_e5-base-v2.jsonl \
    --feedback_path data/dl2019/rewrite/gprf_e5-base-v2_Llama3B.jsonl \
    --feedback_topk 10 \
    --topk 100 \
    --retriever e5-base-v2 \
    --retrieval_method intfloat/e5-base-v2 \
    --pooling_method mean \
    --split test \
    --feedback_is_query
```

```bash
python src/eval.py \
    --qrels_path BEIR/dl2019/qrels/test.tsv \
    --result_path data/dl2019/result/GPRF_bm25.jsonl

python src/eval.py \
    --qrels_path BEIR/dl2019/qrels/test.tsv \
    --result_path data/dl2019/result/GPRF_e5-base-v2.jsonl
```

