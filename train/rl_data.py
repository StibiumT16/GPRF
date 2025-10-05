import json
import pandas as pd
from tqdm import tqdm
import random 

sample_size = 200000
result_topk= 10
models = ['bm25', 'e5-base-v2']
base_path = "../BEIR"

random.seed(42)

system_prompt = "Please rewrite the user's query based on several relevant passages (which may contain noise or errors). The rewritten query should preserve the original meaning while incorporating as much information as possible, so that search engines can more effectively retrieve relevant passages."
user_prompt = "Relevant Passages:\n{documents}\nUser Query: {question}\n\nRewritten Query:"


def format_reference(retrieval_result):
    format_reference = ""
    for idx, content in enumerate(retrieval_result):
        format_reference += f"Passage {idx+1}: {content}\n"

    return format_reference

doc_dict = {}
with open(f'{base_path}/msmarco/corpus.jsonl') as fr:
    for line in tqdm(fr, desc="Load Corpus"):
        line = json.loads(line)
        doc_dict[str(line['id'])] = line['contents']

query_dict = {}
with open(f'{base_path}/msmarco/queries.jsonl') as fr:
    for line in tqdm(fr,desc="Load Query"):
        line = json.loads(line)
        query_dict[str(line['_id'])] = line['text']

result_dict_list = []
for i, model in enumerate(models):
    result_dict = {}
    with open(f'../data/msmarco/original/train_{model}.jsonl') as fr:
        for line in tqdm(fr, desc="Load Retrieval Result"):
            line = json.loads(line)
            result_dict[str(line['qid'])]  = line['results'][:result_topk]
    result_dict_list.append(result_dict)

qrels_df = pd.read_csv(f'{base_path}/msmarco/qrels/train.tsv', sep='\t')
qids = qrels_df['query-id'].unique()

random.shuffle(qids)
qids = qids[:sample_size]

datas = []

for qid in tqdm(qids, desc="Processing"):
    qid = str(qid)
    id = random.randint(0, 1) # 0 / 1
    model = models[id]
    query = query_dict[qid]
    
    
    subset = qrels_df[qrels_df["query-id"] == int(qid)][["corpus-id", "score"]]
    qrel = dict(zip(
        subset["corpus-id"].astype(str),
        subset["score"]
    ))

    
    input_params = {
        "question": query,  
        "documents": format_reference([doc_dict[str(it)] for it in result_dict_list[id][qid]])
    }
    
    datas.append({
        "messages" : [
            {"role": "system", "content": system_prompt.format(**input_params)},
            {"role": "user", "content": user_prompt.format(**input_params)},
        ],
        "original_query" : query,
        "retriever": model,
        "qrels" : str(qrel)
    })
    

with open('train_data/gprf_rl/train.jsonl', 'w') as fw:
    for it in datas:
        fw.write(json.dumps(it) + '\n')