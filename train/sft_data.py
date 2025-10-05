import json
from tqdm import tqdm
import random 

random.seed(42)

topk = 30000
result_topk= 10
models = ['bm25', 'e5-base-v2']

system_prompt = "Please rewrite the user's query based on several relevant passages (which may contain noise or errors). The rewritten query should preserve the original meaning while incorporating as much information as possible, so that search engines can more effectively retrieve relevant passages."
user_prompt = "Relevant Passages:\n{documents}\nUser Query: {question}\n\nRewritten Query:"
def format_reference(retrieval_result):
    format_reference = ""
    for idx, content in enumerate(retrieval_result):
        format_reference += f"Passage {idx+1}: {content}\n"

    return format_reference

doc_dict = {}
with open(f'../BEIR/msmarco/corpus.jsonl') as fr:
    for line in tqdm(fr, desc="Load Corpus"):
        line = json.loads(line)
        doc_dict[str(line['id'])] = line['contents']


datas = []


for model in models:
    def load_result(result_path, result_topk):
        result_dict = {}
        with open(result_path) as fr:
            for line in tqdm(fr, desc="Load Retrieval Result"):
                line = json.loads(line)
                result_dict[str(line['qid'])]  = line['results'][:result_topk]
        return result_dict
    
    def load_query(query_path):
        query_dict = {}
        with open(query_path) as fr:
            for line in tqdm(fr,desc="Load Query"):
                line = json.loads(line)
                id = str(line['_id'])
                if id not in query_dict: query_dict[id] = {}
                query_dict[id][line['num']] = line['text']
        return query_dict

    
    with open(f'../data/train/{model}_rej.jsonl') as fr:
        lst = []
        for line in tqdm(fr, desc="Load Reject Sample Result"):
            line = json.loads(line)
            lst.append((str(line['_id']), int(line['num']), line['new_ndcg'] - line['old_ndcg']))
            
    sorted_lst = sorted(lst, key=lambda x: x[2], reverse=True)[:topk]
    print(sorted_lst[-1][2])
    query_dict = load_query(f'../data/train/{model}_queries.jsonl')
    result_dict = load_result(f'../data/msmarco/original/train_{model}.jsonl', result_topk)
    
    for (qid, num, _) in tqdm(sorted_lst, desc="Process"):
        original_query = query_dict[qid][0]
        assert num > 0
        new_query = query_dict[qid][num]
        passages = [doc_dict[str(it)] for it in result_dict[qid]]
        
        input_params = {
            "question": original_query,  
            "documents": format_reference(passages)
        }
        
        datas.append({
            "messages" : [
                {"role": "system", "content": system_prompt.format(**input_params)},
                {"role": "user", "content": user_prompt.format(**input_params)},
                {"role": "assistant", "content": new_query}
            ]
        })
        
random.shuffle(datas)
with open('train_data/gprf_sft/train.jsonl', 'w') as fw:
    for it in datas:
        fw.write(json.dumps(it) + '\n')