import numpy as np



def compute_metrics(p, metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions = true_predictions, references = true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



def compute_metrics_finegrained(p, metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions = true_predictions, references = true_labels)
    
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
    return {**results_macro, **results_micro}