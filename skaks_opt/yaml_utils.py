from __future__ import annotations

from typing import Any

import yaml


class _FlowSeq(list):
    pass


def _flow_seq_representer(dumper: yaml.Dumper, data: _FlowSeq):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def flowify(value: Any) -> Any:
    if isinstance(value, list):
        if any(isinstance(item, list) for item in value):
            return [flowify(item) for item in value]
        return _FlowSeq([flowify(item) for item in value])
    if isinstance(value, dict):
        return {key: flowify(val) for key, val in value.items()}
    return value


def dump_yaml(data: Any, fh, *, sort_keys: bool | None = None) -> None:
    yaml.add_representer(_FlowSeq, _flow_seq_representer, Dumper=yaml.SafeDumper)
    payload = flowify(data)
    yaml.safe_dump(payload, fh, sort_keys=sort_keys)
