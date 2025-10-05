import json
import warnings
import operator
from functools import reduce
from typing import List
import faiss
import torch
import numpy as np
from tqdm import tqdm
from .utils import load_corpus, load_docs, pooling, load_model


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                if "zh" in self.model_name.lower():
                    query_list = [f"为这个句子生成表示以用于检索相关文章：{query}" for query in query_list]
                else:
                    query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]
                    
        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            if 'pooler_output' in output:
                query_emb = pooling(output.pooler_output,
                                    output.last_hidden_state,
                                    inputs['attention_mask'],
                                    self.pooling_method)
            else:
                query_emb = pooling(None,
                                    output.last_hidden_state,
                                    inputs['attention_mask'],
                                    self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)
    

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        
        if config.rm3:
            print("Run BM25 with RM3")
            self.searcher.set_rm3()
            
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        try:
            hits = self.searcher.search(query, num)
        except:
            cut_query = " ".join(query.split(' ')[:896])
            hits = self.searcher.search(cut_query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'id': hit.docid,
                    'contents': content
                } 
                for hit, content in zip(hits, all_contents)
            ]
        else:
            # results = load_docs(self.corpus, [hit.docid for hit in hits])
            # only need id here
            results = [{'id' : hit.docid} for hit in hits]

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in tqdm(query_list, desc='Retrieval process: '):
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


    def search_with_feedback(self, query: str, feedback: List[str], num: int = None, return_score: bool = False, is_query: bool = False):
        if is_query: 
            new_query = ' '.join([query] + feedback)
        else:
            new_query = ' '.join([f"{query} {content}" for content in feedback])
        return self._search(new_query, num, return_score)        
        
        
    def batch_search_with_feedback(self, query_list: str, feedback_list: List[List[str]], num: int = None, return_score: bool = False, is_query: bool = False):
        results = []
        scores = []
        for query, feedback in tqdm(zip(query_list, feedback_list), desc='Retrieval process: '):
            item_result, item_score = self.search_with_feedback(query, feedback, num, return_score, is_query)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print("Begin Loading Faiss Index:")
        self.index = faiss.read_index(self.index_path)
        print("Finish Loading Faiss Index!")
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size


    def _search(self, query: str, num: int = None, return_score: bool = False):
        return self._batch_search([query], num, return_score)
        '''
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results, None
        '''

    def _batch_search(self, query_list, num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results, scores = [], []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
        
        
        if return_score:
            return results, scores
        else:
            return results, None
    
    def search_with_feedback(self, query: str, feedback: List[str], num: int = None, return_score: bool = False, is_query: bool = False):
        return self.batch_search_with_feedback([query], [feedback], num, return_score, is_query)
        
        
    def batch_search_with_feedback(self, query_list: str, feedback_list: List[List[str]], num: int = None, return_score: bool = False, is_query: bool = False):
        if num is None:
            num = self.topk
        
        results, scores = [], []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            query_emb = self.encoder.encode(query_batch) # [B, D]

            feedback_batch_list = feedback_list[start_idx:start_idx + self.batch_size] # [B, K]
            feedback_batch_list = reduce(operator.concat, feedback_batch_list) # B * K        
            feedback_emb_list = []    
            for start_idx_feedback in range(0, len(feedback_batch_list), self.batch_size):
                feedback_batch = feedback_batch_list[start_idx_feedback:start_idx_feedback+self.batch_size]
                feedback_batch_emb = self.encoder.encode(feedback_batch, is_query=is_query)
                feedback_emb_list.append(feedback_batch_emb)
            feedback_emb = np.vstack(feedback_emb_list).reshape(query_emb.shape[0], -1, query_emb.shape[-1]) # [B, K, D] VPRF

            batch_emb = np.concatenate([feedback_emb, query_emb[:, np.newaxis, :]], axis=1).mean(axis=1)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
        
        
        if return_score:
            return results, scores
        else:
            return results, None

def get_retriever(config):
    if "bm25" in config.retrieval_method.lower():
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


class RetrieverConfig:
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        rm3: bool = False,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.rm3 = rm3

