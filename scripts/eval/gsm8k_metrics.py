"""GSM8K response processing shared with the repository's lm-eval adapter."""


def truncate_response(response, until, eos_token):
    """Apply the adapter's text stops before answer extraction."""
    for stop in [*until, "<|eot_id|>", eos_token]:
        if stop:
            response = response.split(stop)[0]
    return response


def prepare_gsm8k_response(response):
    """Convert the FIRST box to #### exactly as harness_eval.generate_until does.

    Deliberately use string splitting: nested boxes, repeated answers, whitespace,
    and numeric formatting must retain the adapter's existing behavior.
    """
    predicted_answer = None
    if "boxed{" in response:
        predicted_answer = response.split("boxed{")[1].split("}")[0]
        response = response.split("boxed{")[0] + "#### " + predicted_answer
        response = response.replace("$\\", "")
    return response, predicted_answer


def boxed_answer_accuracy(predicted_answer, gold):
    """The adapter's printed Accuracy, distinct from its final lm-eval table."""
    return float(
        predicted_answer is not None and gold.split("### ")[1] == predicted_answer
    )


class GSM8KHarnessScorer:
    """Use the installed harness's GSM8K config, filters, and exact-match metric.

    This loads no model or dataset and downloads nothing. Keep lm-eval imports
    lazy so training and the response helpers do not require the harness.
    """

    primary_metric = "exact_match,strict-match"

    def __init__(self):
        from importlib.metadata import version
        from importlib.resources import files

        import yaml
        from lm_eval.api.metrics import exact_match_fn
        from lm_eval.filters import build_filter_ensemble

        config_path = files("lm_eval") / "tasks/gsm8k/gsm8k.yaml"
        config = yaml.safe_load(config_path.read_text())
        # Fail explicitly if a new task definition needs different processing.
        if config["doc_to_target"] != "{{answer}}":
            raise ValueError("Unsupported lm-eval GSM8K target; review metric parity")
        (metric_config,) = config["metric_list"]
        if (
            metric_config["metric"] != "exact_match"
            or metric_config["aggregation"] != "mean"
        ):
            raise ValueError("Unsupported lm-eval GSM8K metric; review metric parity")
        self.metric = exact_match_fn
        self.metric_kwargs = {
            key: value
            for key, value in metric_config.items()
            if key not in {"metric", "aggregation", "higher_is_better"}
        }
        self.filters = [
            build_filter_ensemble(
                entry["name"],
                [
                    (
                        step["function"],
                        {
                            key: value
                            for key, value in step.items()
                            if key != "function"
                        },
                    )
                    for step in entry["filter"]
                ],
            )
            for entry in config["filter_list"]
        ]
        self.metric_names = [f"exact_match,{f.name}" for f in self.filters]
        if self.primary_metric not in self.metric_names:
            raise ValueError("lm-eval GSM8K strict-match filter is missing")
        strict_filter = next(f for f in self.filters if f.name == "strict-match")
        self.invalid_answer = strict_filter.filters[0]().fallback
        self.until = config["generation_kwargs"]["until"]
        self.metadata = {
            "lm_eval_version": version("lm_eval"),
            "task": config["task"],
            "task_version": config.get("metadata", {}).get("version"),
            "metric_list": config["metric_list"],
            "filter_list": config["filter_list"],
            "until": self.until,
            "primary_metric": self.primary_metric,
        }

    def score(self, completion, gold, eos_token=None):
        from types import SimpleNamespace

        response = truncate_response(completion, self.until, eos_token)
        response, boxed_answer = prepare_gsm8k_response(response)
        instance = SimpleNamespace(
            resps=[response], doc={"answer": gold}, filtered_resps={}
        )
        scores = {"harness_boxed_accuracy": boxed_answer_accuracy(boxed_answer, gold)}
        for pipeline in self.filters:
            pipeline.apply([instance])
            scores[f"exact_match,{pipeline.name}"] = float(
                self.metric(
                    predictions=[instance.filtered_resps[pipeline.name]],
                    references=[gold],
                    **self.metric_kwargs,
                )["exact_match"]
            )
        return {
            "harness_response": response,
            "filtered_resps": instance.filtered_resps,
            "scores": scores,
            "parse_failure": (
                instance.filtered_resps["strict-match"] == self.invalid_answer
            ),
        }
