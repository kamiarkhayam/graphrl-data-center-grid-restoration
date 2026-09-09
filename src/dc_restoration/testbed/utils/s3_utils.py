from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests

S3_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _list_request(base_url: str, prefix: str, delimiter: str | None = None) -> ET.Element:
    query = f"{base_url}?list-type=2&prefix={quote(prefix, safe='/')}"
    if delimiter is not None:
        query += f"&delimiter={quote(delimiter)}"
    response = requests.get(query, timeout=60)
    response.raise_for_status()
    return ET.fromstring(response.text)


def list_common_prefixes(base_url: str, prefix: str, delimiter: str = "/") -> list[str]:
    root = _list_request(base_url, prefix, delimiter)
    return [
        elem.find("s3:Prefix", S3_XML_NS).text
        for elem in root.findall("s3:CommonPrefixes", S3_XML_NS)
    ]


def list_objects(base_url: str, prefix: str) -> list[dict[str, Any]]:
    root = _list_request(base_url, prefix, None)
    records: list[dict[str, Any]] = []
    for obj in root.findall("s3:Contents", S3_XML_NS):
        key = obj.find("s3:Key", S3_XML_NS).text
        size = int(obj.find("s3:Size", S3_XML_NS).text)
        last_modified = obj.find("s3:LastModified", S3_XML_NS).text
        records.append({"key": key, "size": size, "last_modified": last_modified})
    return records


def download_s3_object(
    base_url: str, key: str, destination: str | Path, overwrite: bool = False
) -> Path:
    dest_path = Path(destination)
    if dest_path.exists() and not overwrite:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{base_url}/{quote(key, safe='/')}", stream=True, timeout=120)
    response.raise_for_status()
    with dest_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    return dest_path
