import numpy as np
import evaluate


class EvaluationMetric:
    """
    Caller for computing metrics.
    """
    def __init__(self, id2label: dict[int, str]):
        self.id2label = id2label
        self.metric = evaluate.load("seqeval")
    
    def __call__(self, p) -> dict[str, int | float]:
        probas, true_ids = p
        pred_ids = np.argmax(probas, axis = 2)

        # Remove ignored index (special tokens)
        pred_labels = [
            [self.id2label[p] for (p, t) in zip(preds, trues) if t != -100]
            for preds, trues in zip(pred_ids, true_ids)
        ]
        true_labels = [
            [self.id2label[t] for (p, t) in zip(preds, trues) if t != -100]
            for preds, trues in zip(pred_ids, true_ids)
        ]
        results = self.metric.compute(predictions = pred_labels, references = true_labels)
        
        keys_micro = [k for k, v in results.items() if isinstance(v, dict)]
        results_micro = {
            '{} {}'.format(k, m): results[k][m]
            for k in keys_micro
            for m in ['precision', 'recall', 'f1']
        }
        results_macro = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        return results_macro | results_micro
