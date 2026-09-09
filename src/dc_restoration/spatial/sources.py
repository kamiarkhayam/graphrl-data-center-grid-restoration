"""Explicit downloads and checksum verification of the version-3 sources.

JavaScript is treated only as text. Changed assets or datasets require review
and a new manifest, never an automatic migration to different scientific inputs.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from importlib.resources import files

import numpy as np
import pandas as pd
import requests

EPRI_ASSET = "epri/index.html-BnOKIAPk.js"
METRICS = ["annual_TWh", "peak_GW", "nominal_GW"]
STATES = set(
    "AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY".split()
)


def source_manifest():
    return json.loads(
        files("dc_restoration.spatial").joinpath("source_manifest.json").read_text(encoding="utf-8")
    )


def _check(payload, record, name):
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(
            f"Source checksum mismatch: {name}. The source changed or the cache is corrupt; use the version-3 cache or review the source manifest."
        )


def verify_cache(cache_dir):
    records = source_manifest()
    for name, record in records.items():
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing map source {name}; run dc-fetch-intro-map-sources or supply the archived cache"
            )
        _check(path.read_bytes(), record, name)
    return records


def fetch_sources(config, *, offline=False, session=None):
    cache = config["cache_dir"]
    if offline:
        records = verify_cache(cache)
        extract_epri((cache / EPRI_ASSET).read_text(encoding="utf-8"))
        return records
    records, receipts = source_manifest(), {}
    receipt_path = cache / "source_manifest.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    client = session or requests.Session()
    try:
        for name, record in records.items():
            path = cache / name
            receipt = dict(record)
            if path.exists():
                _check(path.read_bytes(), record, name)
                if name in previous:
                    receipt = dict(previous[name])
                    if receipt.get("sha256") != record["sha256"]:
                        raise ValueError(
                            f"Cached retrieval receipt disagrees with the pinned source: {name}"
                        )
            else:
                try:
                    response = client.get(
                        record["url"],
                        timeout=90,
                        headers={
                            "User-Agent": "Academic introduction-map research; Python requests"
                        },
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise RuntimeError(
                        f"Cannot fetch pinned source {name}; the endpoint or dashboard asset may have changed. Use the archived cache; no replacement dataset was selected."
                    ) from exc
                payload = response.content
                _check(payload, record, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                receipt["url"] = response.url
                receipt["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
            receipts[name] = receipt
        extract_epri((cache / EPRI_ASSET).read_text(encoding="utf-8"))
        (cache / "source_manifest.json").write_text(
            json.dumps(receipts, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if session is None:
            client.close()
    return receipts


def validate_epri(frame):
    required = {"scenario", "state", "year", *METRICS}
    if set(frame) != required or frame.empty:
        raise ValueError("EPRI dataset has an unexpected schema or is empty")
    if frame.duplicated(["scenario", "state", "year"]).any():
        raise ValueError("Duplicate EPRI scenario/state/year")
    expected = {
        (scenario, state, year)
        for scenario, years in [
            ("hist", range(2021, 2025)),
            *[(s, range(2024, 2031)) for s in ("low", "medium", "high")],
        ]
        for state in STATES | {"US"}
        for year in years
    }
    if set(frame[["scenario", "state", "year"]].itertuples(index=False, name=None)) != expected:
        raise ValueError(
            "EPRI dataset dimensions changed; expected the complete 1,275-row 2026 series"
        )
    values = frame[METRICS].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("EPRI demand values must be finite and nonnegative")
    checks = []
    for (scenario, year), group in frame.groupby(["scenario", "year"]):
        for metric in METRICS:
            national = group.loc[group.state.eq("US"), metric].iloc[0]
            state_sum = group.loc[group.state.ne("US"), metric].sum()
            if abs(national - state_sum) > 0.026:
                raise ValueError("EPRI national/state sums exceed the published rounding tolerance")
            checks.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "metric": metric,
                    "published_US": national,
                    "sum_of_states": state_sum,
                    "rounding_difference": state_sum - national,
                }
            )
    return checks


def extract_epri(text):
    arrays = list(re.finditer(r"As=(\[\{scenario:.*?\}\]),La=As", text, re.DOTALL))
    if len(arrays) != 1:
        raise ValueError(
            "EPRI asset structure changed: expected exactly one As data array; JavaScript is never executed"
        )
    array = arrays[0][1]
    pattern = r'\{scenario:"(\w+)",state:"(\w+)",year:(\d+),nominal_GW:([\d.]+),peak_GW:([\d.]+),annual_TWh:([\d.]+)\}'
    matches = list(re.finditer(pattern, array))
    if not matches or array != "[" + ",".join(m[0] for m in matches) + "]":
        raise ValueError("Unparsed EPRI array content; refusing incomplete or modified data")
    frame = pd.DataFrame(
        [
            {
                "scenario": m[1],
                "state": m[2],
                "year": int(m[3]),
                "nominal_GW": float(m[4]),
                "peak_GW": float(m[5]),
                "annual_TWh": float(m[6]),
            }
            for m in matches
        ]
    )
    validate_epri(frame)
    return frame
