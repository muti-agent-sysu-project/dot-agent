"""Post-run adapter for EC2 sensitive_items.json. Keep grader data local."""
import json

from .privacy_ledger import PrivacyLedger


def annotation_ledger(document: dict) -> PrivacyLedger:
    units = {}
    for item in document["items"]:
        for field in item["sensitive_fields"]:
            unit_id = json.dumps([item["entity_id"], field["field"]], ensure_ascii=False)
            if unit_id in units:
                raise ValueError("duplicate annotation")
            units[unit_id] = field["value"]
    return PrivacyLedger(units)
