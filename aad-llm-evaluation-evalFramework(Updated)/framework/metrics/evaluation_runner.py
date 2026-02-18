class EvaluationRunner:
    def __init__(self, test_cases=None, metrics=None):
        self.test_cases = test_cases or []
        self.metrics = metrics or []

    def run(self):
        results = []

        for case in self.test_cases:
            case_dict = case if isinstance(case, dict) else case.to_dict()
            metric_outputs = {}

            for metric in self.metrics:
                metric_outputs[metric.name] = metric.evaluate(**case_dict)

            results.append({
                "input": case_dict["input"],
                "actual_output": case_dict["actual_output"],
                "metrics": metric_outputs
            })

        return results
