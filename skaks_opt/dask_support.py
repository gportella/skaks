from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from dask.distributed import Client, LocalCluster  # type: ignore
except Exception:  # pragma: no cover
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore


def _normalize_scheduler_address(address: str) -> str:
    address = address.strip()
    if not address:
        return address
    if "://" in address:
        return address
    return f"tcp://{address}"


def _modernize_jobqueue_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        updated: Dict[str, Any] = {}
        for key, value in obj.items():
            updated[key] = _modernize_jobqueue_config(value)
        if "job_extra" in updated and "job_extra_directives" not in updated:
            updated["job_extra_directives"] = updated.pop("job_extra")
        else:
            updated.pop("job_extra", None)
        return updated
    if isinstance(obj, list):
        return [_modernize_jobqueue_config(item) for item in obj]
    return obj


def _load_jobqueue_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    text = path.read_text(encoding="utf-8")
    loader = json.loads if path.suffix.lower() == ".json" else None
    if loader is None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for dask-jobqueue") from exc
        loader = yaml.safe_load
    payload = loader(text)
    payload = _modernize_jobqueue_config(payload)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(
            f"dask-jobqueue config {path} must describe a mapping of keyword arguments"
        )
    return payload  # type: ignore[return-value]


def _dask_probe_workers(client: Any) -> int:
    try:
        info = client.scheduler_info()
        workers = info.get("workers", {})
        return len(workers or {})
    except Exception:
        return 0


def _ensure_dask_ready(
    client: Any, *, min_workers: int = 1, timeout: float = 60.0
) -> None:
    if min_workers <= 0:
        min_workers = 1
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _dask_probe_workers(client) >= min_workers:
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"Dask client did not reach {min_workers} worker(s) within {timeout:.0f}s"
    )


@contextlib.contextmanager
def dask_client_from_args(
    args: Any,
) -> "contextlib.Iterator[Tuple[Optional[Any], Optional[int]]]":
    wants_client = any(
        [
            getattr(args, "dask_scheduler", None),
            getattr(args, "dask_workers", None),
            getattr(args, "dask_jobqueue", False),
        ]
    )
    if not wants_client:
        yield None, None
        return

    try:
        from dask_jobqueue import SLURMCluster  # type: ignore
    except Exception:
        SLURMCluster = None  # type: ignore

    cluster: Optional[Any] = None
    client: Optional[Any] = None
    stack = contextlib.ExitStack()
    try:
        if getattr(args, "dask_jobqueue", False):
            if SLURMCluster is None:
                raise SystemExit(
                    "dask-jobqueue must be installed to use --dask-jobqueue"
                )
            config = _load_jobqueue_config(args.dask_jobqueue_config)
            cluster = SLURMCluster(**config)
            jobs = getattr(args, "dask_jobqueue_jobs", None)
            if jobs:
                cluster.scale(jobs)
            adapt_kwargs: Dict[str, Any] = {}
            adapt_min = getattr(args, "dask_jobqueue_adapt_min", None)
            adapt_max = getattr(args, "dask_jobqueue_adapt_max", None)
            if adapt_min is not None:
                adapt_kwargs["minimum"] = adapt_min
            if adapt_max is not None:
                adapt_kwargs["maximum"] = adapt_max
            if adapt_kwargs:
                cluster.adapt(**adapt_kwargs)
            client = Client(cluster)
        elif getattr(args, "dask_scheduler", None):
            address = _normalize_scheduler_address(args.dask_scheduler)
            client = Client(address)
        else:
            workers = getattr(args, "dask_workers", None)
            if not workers:
                raise SystemExit("--dask-workers must be set for local Dask cluster")
            threads = getattr(args, "dask_threads", None) or 1
            cluster = LocalCluster(
                n_workers=max(1, int(workers)),
                threads_per_worker=max(1, int(threads)),
                processes=True,
                dashboard_address=None,
            )
            client = Client(cluster)

        assert client is not None
        stack.callback(lambda: client.close())
        if cluster is not None:
            stack.callback(lambda: cluster.close())

        shard_request = getattr(args, "dask_shards", None)
        min_workers = shard_request if shard_request else 1
        _ensure_dask_ready(client, min_workers=max(1, min_workers))

        shard_hint = (
            shard_request if shard_request else max(1, _dask_probe_workers(client))
        )
        yield client, shard_hint
    finally:
        stack.close()
