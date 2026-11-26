"""One metered, synthetic short probe per endpoint before development sampling."""
import argparse
import json
from pathlib import Path

from scripts.finqa_pair1 import Accounting, endpoints
from eval import collab_experiment_runner as R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    meter = Accounting(root, pair2=True)
    results = []
    with (root/"probe.json").open("x") as out, endpoints(meter) as pair:
        for endpoint in pair:
            wire = json.loads(endpoint.candidate(R.Action.ADVISE, "probe", {}, "Reply with OK.").payload)
            wire["max_tokens"] = 16
            content, receipt = endpoint.call(R.encode(wire), "probe-"+endpoint.role)
            results.append(dict(model=wire["model"], role=endpoint.role, content=content,
                                raw_input=receipt.raw_input, cache_read=receipt.cache_read,
                                output=receipt.output, usd=str(meter.charges[receipt.attempt_id])))
        json.dump(results,out,indent=2)


if __name__ == "__main__":
    main()
