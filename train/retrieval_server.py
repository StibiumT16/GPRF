from typing import List, Optional
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ..src.module import RetrieverConfig, get_retriever

parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
parser.add_argument("--index_path", type=str, default="../data/msmarco/index/e5-base-v2_Flat.index", help="Corpus indexing file.")
parser.add_argument("--corpus_path", type=str, default="BEIR/msmarco/format_corpus.jsonl", help="Local corpus file.")
parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
parser.add_argument("--retrieval_method", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--pooling_method", type=str, default="mean", choices=['cls', 'mean', 'pooler'])
parser.add_argument("--port", type=int, default=11451)
parser.add_argument("--post", type=str, default='retrieve')
args = parser.parse_args()


#####################################
# FastAPI server below
#####################################

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# 1) Build a config (could also parse from arguments).
#    In real usage, you'd parse your CLI arguments or environment variables.
config = RetrieverConfig(
    retrieval_method = args.retrieval_method,  
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    retrieval_topk=args.topk,
    faiss_gpu=True,
    retrieval_model_path=args.retrieval_method,
    retrieval_pooling_method=args.pooling_method,
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
)

# 2) Instantiate a global retriever so it is loaded once and reused.
retriever = get_retriever(config)

@app.post(f"/{args.post}")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return resp


if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.port)
