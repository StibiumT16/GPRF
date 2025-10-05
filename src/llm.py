import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from module import GeneratorConfig, get_generator, HyDEPromptTemplate, LameRPromptTemplate, CoTPromptTemplate, GPRFPromptTemplate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default="BEIR/msmarco/qrels/test.tsv")
    parser.add_argument("--queries_path", type=str, default="BEIR/msmarco/queries.jsonl")
    parser.add_argument("--corpus_path", type=str, default="BEIR/msmarco/corpus.jsonl")
    parser.add_argument("--result_path", type=str, default="/data/msmarco/original/test_e5-base-v2.jsonl")
    parser.add_argument("--output_path", type=str, default="data/msmarco/rewrite/hyde.jsonl")
    parser.add_argument("--result_topk", type=int, default=5)
    
    parser.add_argument("--generator_model", type=str, default="Llama-3.2-3B-Instruct")
    parser.add_argument("--generator_model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--generator_max_input_len", type=int, default=2048)
    parser.add_argument("--generator_batch_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--generator_lora_path", type=str, default=None)

    parser.add_argument("--gpu_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=None)

    #generation params
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    
    return args


def load_qids_and_queries(qrels_path, queries_path, sample_size=None):
    def load_queries(queries_path):
        query_dict = {}
        with open(queries_path) as fr:
            for line in fr:
                line = json.loads(line)
                query_dict[line['_id']] = line['text']
        return query_dict

    query_dict = load_queries(queries_path)
    qrels_df = pd.read_csv(qrels_path, sep='\t')
    qids = qrels_df['query-id'].unique().tolist()
    
    if sample_size:
        random.shuffle(qids)
        qids = qids[:sample_size]
        
    queries = [query_dict[str(id)] for id in qids]
    
    return qids, queries

def load_result_list(qids, corpus_path, result_path, result_topk):
    
    def load_corpus(corpus_path):
        doc_dict = {}
        with open(corpus_path) as fr:
            for line in fr:
                line = json.loads(line)
                doc_dict[line['id']] = line['contents']
        return doc_dict
    
    def load_result(result_path, result_topk):
        result_dict = {}
        with open(result_path) as fr:
            for line in fr:
                line = json.loads(line)
                result_dict[line['qid']]  = line['results'][:result_topk]
        return result_dict
    
    doc_dict = load_corpus(corpus_path)
    result_dict = load_result(result_path, result_topk)
    results = [[doc_dict[it] for it in result_dict[str(qid)]] for qid in qids]
    
    return results


def HyDE(config, args):
    model = get_generator(config)
    
    qids, queries = load_qids_and_queries(args.qrels_path, args.queries_path)
    prompt_template = HyDEPromptTemplate(args.dataset, config)
    input_prompts = [prompt_template.get_string(question=q) for q in tqdm(queries)]
    
    preds = model.generate(input_prompts)   
    
    with open(args.output_path, 'w') as fw:
        for qid, pred in zip(qids, preds):
            fw.write(json.dumps({
                'id' : qid,
                'feedback' : pred
            }) + '\n')


def CoT(config, args):
    model = get_generator(config)
    qids, queries = load_qids_and_queries(args.qrels_path, args.queries_path)
    prompt_template = CoTPromptTemplate(config)
    input_prompts = [prompt_template.get_string(question=q) for q in tqdm(queries)]
    
    preds = model.generate(input_prompts)  
    
    with open(args.output_path, 'w') as fw:
        for qid, query, pred in zip(qids, queries, preds):
            text = f"{query} {query} {query} {query} {query} {pred}"
            fw.write(json.dumps({
                '_id' : qid,
                'text' : text
            }) + '\n')
    
    
def LameR(config, args):
    model = get_generator(config)
    
    qids, queries = load_qids_and_queries(args.qrels_path, args.queries_path)
    results = load_result_list(qids, args.corpus_path, args.result_path, args.result_topk)
    prompt_template = LameRPromptTemplate(args.dataset, config)
    
    input_prompts = [prompt_template.get_string(question=q, passages=result) for q,result in tqdm(zip(queries, results))]
    
    preds = model.generate(input_prompts)   
    
    with open(args.output_path, 'w') as fw:
        for qid, query, pred in zip(qids, queries, preds):
            text = ' '.join([f"{query} {passage}" for passage in pred])
            fw.write(json.dumps({
                '_id' : qid,
                'text' : text
            }) + '\n')


def QRG(config, args):
    model = get_generator(config)
    
    qids, queries = load_qids_and_queries(args.qrels_path, args.queries_path, args.sample_size)
    results = load_result_list(qids, args.corpus_path, args.result_path, args.result_topk)
    prompt_template = GPRFPromptTemplate(config)

    
    input_prompts = [prompt_template.get_string(question=q, passages=result) for q,result in tqdm(zip(queries, results))]
    
    preds = model.generate(input_prompts)  
    
    with open(args.output_path, 'w') as fw:
        for qid, q, pred in zip(qids, queries, preds):
            fw.write(json.dumps({
                '_id' : qid,
                'num' : 0,
                'text' : q
            }) + '\n')
            for i, query in enumerate(pred):
                fw.write(json.dumps({
                '_id' : qid,
                'num' : i + 1,
                'text' : query
            }) + '\n')


def GPRF(config, args):
    model = get_generator(config)
    
    qids, queries = load_qids_and_queries(args.qrels_path, args.queries_path, args.sample_size)
    results = load_result_list(qids, args.corpus_path, args.result_path, args.result_topk)
    prompt_template = GPRFPromptTemplate(config)

    input_prompts = [prompt_template.get_string(question=q, passages=result) for q,result in tqdm(zip(queries, results))]
    
    preds = model.generate(input_prompts)  
    
    with open(args.output_path, 'w') as fw:
        for qid, pred in zip(qids, preds):
            fw.write(json.dumps({
                'id' : qid,
                'feedback' : pred
            }) + '\n')
    

if __name__ == '__main__':    
    args = parse_args()
    
    generation_params = {'do_sample' : args.do_sample, 'max_tokens' : args.max_tokens}
    if args.n:
        generation_params['n'] = args.n
    if args.do_sample:
        if args.temperature:
            generation_params['temperature'] = args.temperature
        if args.top_p:
            generation_params['top_p'] = args.top_p
    
    print(generation_params)
    
    config = GeneratorConfig(
        generator_model=args.generator_model,
        generator_model_path=args.generator_model_path,
        gpu_num=args.gpu_num,
        seed=args.seed,
        generator_max_input_len=args.generator_max_input_len,
        generator_batch_size=args.generator_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generator_lora_path=args.generator_lora_path,
        generation_params=generation_params
    )
    
    if args.method.lower() == 'hyde':
        HyDE(config, args)
    elif args.method.lower() == 'cot':
        CoT(config, args)
    elif args.method.lower() == 'lamer':
        LameR(config, args)
    elif args.method.lower() == "qrg":
        QRG(config, args)
    elif args.method.lower() == "gprf":
        GPRF(config, args)