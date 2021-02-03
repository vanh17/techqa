import collections
import json
import logging
import sys

import argparse
from typing import Dict, List, Tuple, Optional, Union

_DEFAULT_TOP_K = 5


class EVAL_OPTS():
    def __init__(self, data_file, pred_file, top_k=5):
        self.data_file = data_file
        self.pred_file = pred_file
        self.top_k = top_k


OPTS = EVAL_OPTS(data_file=None, pred_file=None)

ScoresById = Dict[str, Union[int, float]]
TopKScoresById = Dict[str, List[Union[int, float]]]


def parse_args():
    parser = argparse.ArgumentParser(
        """
        Official evaluation script for TechQA v1. It will produce the following metrics: 
        
        - "QA_F1": Calculated for precision/recall based on character offset. The threshold 
        provided in the prediction json will be applied to predict NO ANSWER in cases 
        where the prediction score < threshold.
        
        - "IR_Precision": Calculated based on doc id match. The threshold provided in the 
        prediction json will be applied to predict NO ANSWER in cases where the prediction
         score < threshold.
        
        - "HasAns_QA_F1": Same as `QA_F1`, but calculated only on answerable questions. 
        Thresholds are ignored for this calculation.
        
        - "HasAns_Top_k_QA_F1": The max `QA_F1` based on the top `k` predictions calculated 
        only on answerable questions. Thresholds are ignored for this calculation. 
        By default k=%d.
        
        - "HasAns_IR_Precision": Same as `IR_Precision`, but calculated only on answerable 
        questions. Thresholds are ignored for this calculation.
        
        - "HasAns_Top_k_IR_Precision": The max `IR_Precision` based on the top `k` predictions
         calculated only on answerable questions. Thresholds are ignored for this calculation. 
        By default k=%d.
        
        - "Best_QA_F1": Same as `QA_F1`, but instead of applying the provided threshold, it 
        will scan for the `optimal` threshold based on the evaluation set.
        
        - "Best_QA_F1_Threshold": The threshold identified during the search for `Best_QA_F1`
        
        - "_Total_Questions": All metrics will be accompanied by a `_Total_Questions` count of
         the number of queries used to compute the statistic.
         """ % (_DEFAULT_TOP_K, _DEFAULT_TOP_K))
    parser.add_argument('--data_file', metavar='dev_vX.json',
                        help='Input competition query annotations JSON file.'
                        '''
                        Model data Json file in the format:
                        {"id": 113, "claim": "Angiotensin converting enzyme inhibitors are associated with increased risk for functional renal insufficiency.", 
                        "evidence": {"6157837": [{"sentences": [2], "label": "SUPPORT"}, {"sentences": [7], "label": "SUPPORT"}]}, "cited_doc_ids": [6157837]}
                        {"id": 94, "claim": "Albendazole is used to treat lymphatic filariasis.", "evidence": {}, "cited_doc_ids": [1215116]}
                        {"id": 115, "claim": "Anthrax spores can be disposed of easily after they are dispersed.", 
                        "evidence": {"33872649": [{"sentences": [6], "label": "CONTRADICT"}]}, "cited_doc_ids": [33872649]}

                        ''')
    parser.add_argument('--pred_file', metavar='pred.json',
                        help=
                        """
                        Model predictions sentene JSON file in the format: 
                        {
                        "907": [{"6106004_0":{"doc":0.22488001,"sent":0.14946598}},{"6106004_0":{"doc":0.22488001,"sent":0.14946598}}], 
                        "471":[{"1754001_2":{"doc":0.69075906,"sent":0.29445484}},{"7185591_0":{"doc":0.33188218,"sent":0.20788}}]
                        }
                             
                             """)
    parser.add_argument('--top_k', '-k', type=int, default=_DEFAULT_TOP_K,
                        help='Eval script will compute F1 score using the top 1 prediction'
                             ' as well as the top k predictions')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def compute_metrics_sent(gold_sents, pred_sents):
    overlap = len(list(set(gold_sents) & set(pred_sents)))
    precision = 1.0 * overlap / len(pred_sents)
    recall = 1.0 * overlap / len(gold_sents)
    f1 = 0
    if precision + recall != 0.0:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall


def main(OPTS):
    data = []
    with open(OPTS.data_file, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
        dataset = {str(claim['id']): claim["evidence"] for claim in data}
    with open(OPTS.pred_file, encoding='utf-8') as f:
        system_output = json.load(f)
        preds = system_output

    out_eval = evaluate(preds=preds, dataset=dataset, top_k=OPTS.top_k)
    print(json.dumps(out_eval, indent=2))
    return out_eval


def evaluate(preds: Dict[str, List[Dict]], dataset: Dict[str, Dict], top_k: int):
    f1_score_sum = 0
    recall_score_sum = 0
    total_has_evidence = 1
    total = 1
    claim_ids = preds.keys()
    for claim_id in claim_ids:
        total += 1
        if len(dataset[claim_id].keys()) > 0:
            # Ignore claims without evidence
            total_has_evidence += 1
            # get predicted sentences from preds with top k.
            pred_sents = [list(ps.keys())[0] for ps in preds[claim_id][:top_k]]
            # get gold sentences from dataset
            gold_sents = [(doc + "_" + str(sent_id)) for doc in dataset[claim_id].keys() for sents in dataset[claim_id][doc] for sent_id in sents["sentences"]]
            f1, recall = compute_metrics_sent(gold_sents, pred_sents)
            f1_score_sum += f1
            recall_score_sum += recall
    return collections.OrderedDict([
        ('F1', 100.0 * f1_score_sum / total_has_evidence),
        ('Recall', 100.0 * recall_score_sum / total_has_evidence),
        #('IR_Precision', 100.0 * retrieval_score_sum / total),
        ('Total_Questions_Has_Evidence', total_has_evidence),
        ('Total_Questions', total),
    ])


if __name__ == '__main__':
    OPTS = parse_args()
    main(OPTS)