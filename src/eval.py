import json, os
import argparse
import pandas as pd
import numpy as np
import pytrec_eval


parser = argparse.ArgumentParser()
parser.add_argument("--qrels_path", type=str, default="BEIR/msmarco/qrels/test.tsv")
parser.add_argument("--result_path", type=str, default="data/msmarco/original/test_bm25.jsonl")
args = parser.parse_args()


df = pd.read_csv(args.qrels_path, sep='\t')

qids, qrel, run = [], {}, {}

for qid, did, score in zip(df['query-id'], df['corpus-id'], df['score']):
    qid, did, score = str(qid), str(did), int(score)
    
    if qid not in qrel:
        qrel[qid] = {}   
    qrel[qid][did] = score
    

with open(args.result_path) as fr:
    for line in fr:
        line = json.loads(line)
        qid = str(line['qid'])
        qids.append(qid)
        run[qid] = {}
        for did, score in zip(line['results'], line['scores']):
            run[qid][did] = score

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall_10', 'ndcg_cut.10', 'recall_100', 'ndcg_cut.100'})
result = evaluator.evaluate(run)

r10, r100, ndcg10, ndcg100 = [], [], [], []
for qid in qids:
    r10.append(result[qid]['recall_10'])
    r100.append(result[qid]['recall_100'])
    ndcg10.append(result[qid]['ndcg_cut_10'])
    ndcg100.append(result[qid]['ndcg_cut_100'])

print(f"Result of {args.result_path}:")
print(f"{np.mean(ndcg10):.4f},{np.mean(r100):.4f}")
