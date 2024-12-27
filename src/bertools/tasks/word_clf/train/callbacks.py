from typing import Any

import pandas as pd
from transformers.integrations import ClearMLCallback, MLflowCallback


class CustomClearMLCallback(ClearMLCallback):
    """
    Minical callback class, to send logs to clearml in the desired format.
    """

    def _report_scalars(self, scalars, iteration: int) -> None:
        for scalar in scalars:
            self._clearml_task.get_logger().report_scalar(**scalar, iteration=iteration)
        return

    def _report_tables(self, scores: dict[str, Any], iteration: int, prefix: str) -> None:
        """
        Send values to tables in "Plots" section of ClearML.
        """
        tables = {
            "1 Toxicity detection": {k: scores[k] for k in ["1 Toxicity detection"]},
            "2 Illegality detection": {k: scores[k] for k in ["2 Illegality detection"]},
            "3 Priority detection": scores["3 Priority detection"],
            "4 Span detection": {k: v["cooccur"] for k, v in scores["4 Span detection"].items()},
        }
        for t_name, t in tables.items():
            self._clearml_task.get_logger().report_table(
                title=f"{prefix} scores",
                series=t_name,
                table_plot=pd.DataFrame(t).round(3),
                iteration=iteration,
            )
        return

    def _report_curve(self, figure, iteration: int, prefix: str) -> None:
        """
        Left unused.
        """
        self._clearml_task.get_logger().report_plotly(
            title=f"{prefix} - Recall-FPR curve",
            series="Recall-FPR curve",
            iteration=iteration,
            figure=figure,
        )
        return

    def _convert_report_to_scalars(self, report: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
        """
        Convert report to metrics suitable for ClearML logging.
        """
        measures = ["precision", "recall", "f1-score"]
        return (
            [
                {"title": f"{prefix} - {s}", "series": m, "value": report[s][m]}
                for s in ["1 Toxicity detection", "2 Illegality detection"]
                for m in measures + ["fpr"]
            ]
            + [
                {"title": f"{prefix} - {s} - {m}", "series": prio, "value": vals[m]}
                for s in ["3 Priority detection"]
                for m in measures
                for prio, vals in report[s].items()
            ]
            + [
                {"title": f"{prefix} - {s} - {m} - {method}", "series": label, "value": methods[method][m]}
                for s in ["4 Span detection"]
                for m in measures
                for method in ["cooccur"]
                for label, methods in report[s].items()
            ]
        )

    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs) -> None:
        """
        Push scores to ClearML.
        """
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)

        iteration = state.epoch if args.eval_strategy == "epoch" else state.global_step
        scalars: list[dict[str, Any]] = []
        if state.is_world_process_zero:
            scalars = []
            if v := logs.get("train_loss", None):
                scalars.append({"title": "- Loss -", "series": "train", "value": v})
            if v := logs.get("loss", None):
                scalars.append({"title": "- Loss -", "series": "train", "value": v})
            if v := logs.get("eval_loss", None):
                scalars.append({"title": "- Loss -", "series": "eval", "value": v})
            if report := logs.get("eval_scores", None):
                scalars += self._convert_report_to_scalars(report, prefix="Eval")
                self._report_tables(report, prefix="Eval", iteration=iteration)
            if report := logs.get("test_scores", None):
                self._report_tables(report, prefix="Test", iteration=iteration)

            self._report_scalars(scalars, iteration=iteration)
        return


class CustomMLflowCallback(MLflowCallback):
    """
    Minical callback class, to send logs to MLflow in the desired format.
    """

    def _report_scalars(self, scalars, iteration: int) -> None:
        """
        Log metrics to MLflow.
        """
        self._ml_flow.log_metrics(metrics=scalars, step=iteration, synchronous=False)
        return

    def _convert_report_to_scalars(self, report: dict[str, Any], prefix: str) -> dict[str, Any]:
        """
        Convert report to metrics suitable for MLflow logging.
        """
        measures = ["precision", "recall", "f1-score"]
        return (
            {
                f"{prefix} - {s}/{m}": report[s][m]
                for s in ["1 Toxicity detection", "2 Illegality detection"]
                for m in measures + ["fpr"]
            }
            | {
                f"{prefix} - {s} - {m}/{prio}": vals[m]
                for s in ["3 Priority detection"]
                for m in measures
                for prio, vals in report[s].items()
            }
            | {
                f"{prefix} - {s} - {m} - {method}/{label.replace('/', '-')}": methods[method][m]
                for s in ["4 Span detection"]
                for m in measures
                for method in ["cooccur"]
                for label, methods in report[s].items()
            }
        )

    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs) -> None:
        """
        Push scores to MLflow.
        """
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)

        iteration = state.epoch if args.eval_strategy == "epoch" else state.global_step
        if state.is_world_process_zero:
            scalars = {}
            if v := logs.get("train_loss", None):
                scalars["- Loss -/train"] = v
            if v := logs.get("loss", None):
                scalars["- Loss -/train"] = v
            if v := logs.get("eval_loss", None):
                scalars["- Loss -/eval"] = v
            if report := logs.get("eval_scores", None):
                scalars |= self._convert_report_to_scalars(report, prefix="Eval")

            self._report_scalars(scalars, iteration=iteration)
        return


def remove_logs_from_strout(self, args, state, control, logs=None, **kwargs):
    """
    see https://github.com/huggingface/transformers/issues/18093#issuecomment-1696909538
    """
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
