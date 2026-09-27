"""I/O helpers for TSV submissions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def write_id_map(
    path: Path | str,
    s1_ids: Iterable[str],
    mapping: Mapping[str, Iterable[str]],
    id_col: str,
    list_col: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{id_col}\t{list_col}\n")
        write_id_map_rows(f, s1_ids, mapping)


def write_id_map_rows(f, s1_ids: Iterable[str], mapping: Mapping[str, Iterable[str]]) -> None:
    """Append rows to an already-open TSV file handle (no header)."""
    for s1 in s1_ids:
        ids = list(dict.fromkeys([x for x in mapping.get(s1, []) if x]))
        f.write(f"{s1}\t{','.join(ids)}\n")


def write_matching(path, s1_ids, mapping):
    write_id_map(
        path, s1_ids, mapping, "source1_entity_id", "matched_entity_ids"
    )


def write_candidates(path, s1_ids, mapping):
    write_id_map(
        path, s1_ids, mapping, "source1_entity_id", "candidate_entity_ids"
    )


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    import pandas as pd
    from .metrics import parse_id_list

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        row.source1_entity_id: parse_id_list(row.matched_entity_ids)
        for row in df.itertuples(index=False)
    }
