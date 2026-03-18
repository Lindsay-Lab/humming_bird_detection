#build_yolo.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from humming_bird_detection.configs.config import PrepConfig
from humming_bird_detection.preprocessing.validate import read_yolo_lines, write_yolo_lines

def _load_names_from_yaml(yaml_path: Path) -> Dict[int, str]:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed — cannot parse data.yaml")
        return {}

    try:
        data = yaml.safe_load(yaml_path.read_text())
        names = data.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception as exc:
        logger.debug(f"Failed to parse {yaml_path}: {exc}")
    return {}

def resolve_hummingbird_ids(
    raw_root: Path,
    cfg: PrepConfig,
) -> Set[int]:
    """Return the set of *original* class IDs that represent hummingbirds.

    1. cfg.hummingbird_ids if explicitly provided.
    2. Auto-detect from data.yaml / dataset.yaml in *raw_root*
       by matching cfg.hummingbird_name_hints against class names.

    Raises RuntimeError if no IDs can be resolved.
    """
    if cfg.hummingbird_ids:
        ids = set(cfg.hummingbird_ids)
        logger.info(f"Using explicit hummingbird IDs: {sorted(ids)}")
        return ids

    candidates = [
        raw_root / "data.yaml",
        raw_root / "dataset.yaml",
        raw_root / "data" / "data.yaml",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        id2name = _load_names_from_yaml(candidate)
        hb: Set[int] = set()
        for cid, name in id2name.items():
            if name and any(h in name.lower() for h in cfg.hummingbird_name_hints):
                hb.add(cid)
        if hb:
            logger.info(
                f"Resolved hummingbird IDs from {candidate.name}: {sorted(hb)} "
                f"(full map: {id2name})"
            )
            return hb

    raise RuntimeError(
        "Could not auto-resolve hummingbird class IDs from any data.yaml/dataset.yaml. "
        "Provide PrepConfig(hummingbird_ids={...}) explicitly."
    )

def remap_label_file(
    src_lbl: Path,
    dst_lbl: Path,
    hb_ids: Set[int],
) -> int:
    """Remap classes in *src_lbl* → *dst_lbl*.
    - class ∈ hb_ids  → 0  (hummingbird)
    - class ∉ hb_ids  → 1  (other_bird, e.g. bananaquit)
    Background images have empty label files (no boxes) — no remapping needed.
    Returns number of rows written.
    """
    rows = read_yolo_lines(src_lbl)
    remapped: List[List[float]] = []
    for r in rows:
        orig_c = int(r[0])
        new_c = 0 if orig_c in hb_ids else 1
        remapped.append([float(new_c), r[1], r[2], r[3], r[4]])

    write_yolo_lines(dst_lbl, remapped)
    return len(remapped)

def remap_labels(
    src_label_dir: Path,
    dst_label_dir: Path,
    hb_ids: Set[int],
    num_workers: int = 8,
) -> int:
    """Remap every .txt in *src_label_dir* into *dst_label_dir* (parallel).
    Returns total number of files processed.
    """
    if not src_label_dir.exists():
        logger.warning(f"Source label dir does not exist: {src_label_dir}")
        return 0

    dst_label_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(src_label_dir.glob("*.txt"))
    logger.info(f"Remapping {len(label_files)} labels ({num_workers} workers): hb_ids={sorted(hb_ids)} → 0 (hummingbird), rest → 1 (other_bird)")

    total_boxes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(remap_label_file, lbl, dst_label_dir / lbl.name, hb_ids): lbl
            for lbl in label_files
        }
        for fut in as_completed(futures):
            total_boxes += fut.result()

    logger.info(f"Remapped {len(label_files)} files ({total_boxes} total boxes)")
    return len(label_files)