import ir_measures
import sys


def _get_metrics(metrics):
    measures, errors = [], []
    for m in metrics:
        try:
            measure = ir_measures.parse_measure(m)
            if measure not in measures:
                measures.append(measure)
        except ValueError:
            errors.append(f'syntax error: {m}')
        except NameError:
            errors.append(f'unknown metrics: {m}')
    if errors:
        sys.stderr.write('\n'.join(['error parsing metrics'] + errors + ['']))
        sys.exit(-1)
    return measures


class Evaluator:
    def __init__(self, qrel_file, metrics):
        self.qrels = self._read_qrel(qrel_file)
        self.metrics = _get_metrics(metrics)

    def compute_metrics(self, scores, indices, q_lookup):
        run = {}
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, scores, indices):
            run[qid] = {}
            for docid, score in zip(q_doc_indices, q_doc_scores):
                run[qid][docid] = float(score)

        results = ir_measures.calc_aggregate(self.metrics, self.qrels, run)
        return results.items()

    def _read_qrel(self, qrel_file):
        qrels = {}
        with open(qrel_file, 'r') as f:
            for l in f:
                try:
                    qid, _, docid, rel = l.strip().split('\t')
                except ValueError:
                    raise ValueError("Wrong qrel format.")

                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)
        return qrels
