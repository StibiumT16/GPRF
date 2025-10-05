import json, os
import argparse
import pandas as pd
from tqdm import tqdm
from module import RetrieverConfig, get_retriever


parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
parser.add_argument("--qrels_path", type=str, default="BEIR/msmarco/qrels/test.tsv")
parser.add_argument("--queries_path", type=str, default="BEIR/msmarco/queries.jsonl")
parser.add_argument("--corpus_path", type=str, default="BEIR/msmarco/format_corpus.jsonl")
parser.add_argument("--index_path", type=str, default="data/msmarco/index/e5-base-v2_Flat.index")
parser.add_argument("--feedback_path", type=str, default=None)
parser.add_argument("--feedback_corpus_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default="data/msmarco/original/test_e5-base-v2.jsonl")

parser.add_argument("--topk", type=int, default=10, help="Number of retrieved passages for one query.")
parser.add_argument("--feedback_topk", type=int, default=10)
parser.add_argument("--retriever", type=str, default="e5-base-v2")
parser.add_argument("--retrieval_method", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--pooling_method", type=str, default="mean", choices=['cls', 'mean', 'pooler', 'last'])
parser.add_argument("--split", type=str, default="test", choices=['train', 'dev', 'test'])
parser.add_argument("--rm3", action="store_true", default=False)
parser.add_argument("--feedback_is_query", action="store_true", default=False)
args = parser.parse_args()


query_dict = {}
with open(args.queries_path) as fr:
    for line in fr:
        line = json.loads(line)
        query_dict[str(line['_id'])] = line['text']
        
qrels_df = pd.read_csv(args.qrels_path, sep='\t')
qids = qrels_df['query-id'].unique()

queries = [query_dict[str(id)] for id in qids]


config = RetrieverConfig(
    retrieval_method = args.retriever,  
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    retrieval_topk=args.topk,
    faiss_gpu=True,
    retrieval_model_path=args.retrieval_method,
    retrieval_pooling_method=args.pooling_method,
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
    rm3=args.rm3
)

model = get_retriever(config)

if not args.feedback_path:
    results, scores = model.batch_search(
        query_list=queries,
        num=args.topk,
        return_score=True
    )
else:
    print("=== Run with PRF!!! ===")
    feedbacks = []
    
    if args.feedback_corpus_path:
        feedback_dict = {}
        print("Load feedback corpus...")
        with open(args.feedback_corpus_path) as fr:
            for line in tqdm(fr):
                line = json.loads(line)
                feedback_dict[line['id']] = line['contents']
        print("Finish loading!")
        
        with open(args.feedback_path) as fr:
            for line in fr:
                line = json.loads(line)
                fids = line['results'][:args.feedback_topk]
                feedbacks.append([feedback_dict[fid] for fid in fids])
        
    else:
        with open(args.feedback_path) as fr:
            for line in fr:
                line = json.loads(line)
                feedbacks.append(line['feedback'][:args.feedback_topk])
    
    results, scores = model.batch_search_with_feedback(
        query_list=queries,
        feedback_list=feedbacks,
        num=args.topk,
        return_score=True,
        is_query=args.feedback_is_query
    )


with open(args.output_path, 'w') as fw:
    for qid, result, score in zip(qids, results, scores):
        dids = [it['id'] for it in result]
        
        fw.write(json.dumps({
            'qid' : str(qid),
            'results' : dids,
            'scores' : score
        }) + '\n')
