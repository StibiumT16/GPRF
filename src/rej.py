import json, os
import argparse
import pandas as pd
from tqdm import tqdm
import pytrec_eval
from module import RetrieverConfig, get_retriever


parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
parser.add_argument("--qrels_path", type=str, default="BEIR/msmarco/qrels/test.tsv")
parser.add_argument("--queries_path", type=str, default="data/train/e5-base-b2_queries.jsonl")
parser.add_argument("--corpus_path", type=str, default="BEIR/msmarco/corpus.jsonl")
parser.add_argument("--index_path", type=str, default="data/msmarco/index/e5-base-v2_Flat.index")
parser.add_argument("--output_path", type=str, default="data/train/e5-base-b2_rej.jsonl")

parser.add_argument("--retriever", type=str, default="e5-base-v2")
parser.add_argument("--retrieval_method", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--pooling_method", type=str, default="mean", choices=['cls', 'mean', 'pooler'])
args = parser.parse_args()


qrel, qids, queries = {}, [], []
with open(args.queries_path) as fr:
    for line in fr:
        line = json.loads(line)
        qids.append(f"{line["_id"]}_{line['num']}")
        queries.append(line['text'])

df = pd.read_csv(args.qrels_path, sep='\t')

for qid, did, score in zip(df['query-id'], df['corpus-id'], df['score']):
    qid, did, score = qid, str(did), int(score)
    
    if f"{qid}_0" not in qrel:
        for i in range(11):
            qrel[f'{qid}_{i}'] = {}   
    for i in range(11):
        qrel[f'{qid}_{i}'][did] = score 
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut.10', 'recall_100'})


config = RetrieverConfig(
    retrieval_method = args.retriever,  
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    retrieval_topk=100,
    faiss_gpu=True,
    retrieval_model_path=args.retrieval_method,
    retrieval_pooling_method=args.pooling_method,
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
)

model = get_retriever(config)

results, scores = model.batch_search(
    query_list=queries,
    num=100,
    return_score=True
)

results = [[str(it['id']) for it in result] for result in results]

run = {}
for qid, result, score in tqdm(zip(qids, results, scores)):
    run[qid] = {}
    for did, d_score in zip(result, score):
        run[qid][str(did)] = d_score

eval_result = evaluator.evaluate(run)
eval_result = [(eval_result[qid]['ndcg_cut_10'], eval_result[qid]['recall_100']) for qid in tqdm(qids)]

    
with open(args.output_path, 'w') as fw:
    num_groups = len(eval_result) // 11
    for g in tqdm(range(num_groups)):
        start = g * 11
        end = start + 11
        
        group_results = eval_result[start:end]
        group_qids = qids[start:end]
        
        baseline_ndcg, baseline_recall = group_results[0]
        
        candidates = []
        for i in range(1, 11):
            ndcg, recall = group_results[i]
            if ndcg >= baseline_ndcg and recall >= baseline_recall and not (ndcg == baseline_ndcg and recall == baseline_recall):
                candidates.append((group_qids[i], ndcg, recall))
        
        if candidates:
            best_qid, best_ndcg, best_recall = max(candidates, key=lambda x: (x[1], x[2]))
            fw.write(json.dumps({
                "_id": best_qid.split('_')[0],
                "num" : best_qid.split('_')[1],
                "new_ndcg" : best_ndcg,
                "old_ndcg" : baseline_ndcg
            }) + '\n')
            