import hashlib
import json
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import torch
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
)
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor


_WORD_SPLIT_RE = re.compile(r"\S+")
_TOC_ENTRY_RE = re.compile(r"^(?:\d+(?:\.\d+)*)\s+.+?\.{2,}\s*\d+\s*$")


def _split_words(text: str, max_words: int = 120) -> List[str]:
    words = _WORD_SPLIT_RE.findall(text or "")
    if not words:
        return ["."]
    return words[:max_words]


def _normalize_bbox(bbox: List[float], width: float, height: float) -> List[int]:
    x0, y0, x1, y1 = bbox
    width = max(width, 1.0)
    height = max(height, 1.0)

    def _n(value: float, denom: float) -> int:
        return int(max(0, min(1000, round(1000.0 * value / denom))))

    return [_n(x0, width), _n(y0, height), _n(x1, width), _n(y1, height)]


def _clamp_bbox(bbox: List[float], width: float, height: float) -> List[float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(width - 1.0, x0))
    y0 = max(0.0, min(height - 1.0, y0))
    x1 = max(x0 + 1.0, min(width, x1))
    y1 = max(y0 + 1.0, min(height, y1))
    return [x0, y0, x1, y1]


def _tighten_text_bbox(
    bbox: List[float],
    page_words: List[Tuple[float, float, float, float, str]],
    page_width: float,
    page_height: float,
) -> List[float]:
    """Light tightening: only adjust text blocks with dense word content; skip large/image blocks."""
    if not page_words:
        return _clamp_bbox(bbox, page_width, page_height)

    x0, y0, x1, y1 = [float(v) for v in bbox]
    block_area = (x1 - x0) * (y1 - y0)
    page_area = page_width * page_height
    
    # Skip tightening for very large blocks (likely images or containers).
    if block_area > page_area * 0.25:
        return _clamp_bbox(bbox, page_width, page_height)

    epsilon = 2.0
    contained: List[Tuple[float, float, float, float, str]] = []

    for wx0, wy0, wx1, wy1, wtext in page_words:
        cx = (wx0 + wx1) / 2.0
        cy = (wy0 + wy1) / 2.0
        if (x0 - epsilon) <= cx <= (x1 + epsilon) and (y0 - epsilon) <= cy <= (y1 + epsilon):
            contained.append((wx0, wy0, wx1, wy1, wtext))

    # Short labels are often a single word; allow tightening when we have any
    # contained words, but avoid over-shrinking blocks with no text evidence.
    if not contained:
        return _clamp_bbox(bbox, page_width, page_height)

    tx0 = min(w[0] for w in contained) - 2.0
    ty0 = min(w[1] for w in contained) - 2.0
    tx1 = max(w[2] for w in contained) + 2.0
    ty1 = max(w[3] for w in contained) + 2.0

    return _clamp_bbox([tx0, ty0, tx1, ty1], page_width, page_height)


def _bbox_key(bbox: List[float], precision: int = 1) -> Tuple[float, float, float, float]:
    return tuple(round(float(v), precision) for v in bbox)


def _bbox_intersection_area(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_area(bbox: List[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    inter = _bbox_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(union, 1e-6)


def _normalized_block_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _dedupe_page_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop looser PyMuPDF duplicates: same text with a tight box inside a large container, and overlapping images."""
    text_blocks = [
        b
        for b in page_blocks
        if not b.get("is_image") and _normalized_block_text(str(b.get("text", "")))
    ]
    image_blocks = [b for b in page_blocks if b.get("is_image")]

    to_remove: set[str] = set()

    text_by_area = sorted(text_blocks, key=lambda b: _bbox_area(b["bbox"]))
    for small in text_by_area:
        if small["id"] in to_remove:
            continue
        sa = _bbox_area(small["bbox"])
        if sa < 1.0:
            continue
        ts = _normalized_block_text(str(small.get("text", "")))
        for big in text_by_area:
            if big is small or big["id"] in to_remove:
                continue
            ba = _bbox_area(big["bbox"])
            if ba <= sa * 1.08:
                continue
            inter = _bbox_intersection_area(small["bbox"], big["bbox"])
            if inter < sa * 0.90:
                continue
            tb = _normalized_block_text(str(big.get("text", "")))
            if ts and ts == tb:
                to_remove.add(big["id"])
                continue
            if ts and not tb:
                to_remove.add(big["id"])
                continue
            if ts and tb and ts in tb and len(tb) <= len(ts) * 1.03:
                to_remove.add(big["id"])

    # Drop line-level duplicates that PyMuPDF also returns inside a multiline parent (not TOC lines).
    for small in text_by_area:
        if small["id"] in to_remove:
            continue
        ts = _normalized_block_text(str(small.get("text", "")))
        if not ts or _looks_like_toc_entry(ts) or _looks_like_toc_heading(ts):
            continue
        sa = _bbox_area(small["bbox"])
        if sa < 1.0:
            continue
        for big in text_by_area:
            if big is small or big["id"] in to_remove:
                continue
            raw_big = str(big.get("text", ""))
            if "\n" not in raw_big:
                continue
            ba = _bbox_area(big["bbox"])
            if ba <= sa * 1.75:
                continue
            inter = _bbox_intersection_area(small["bbox"], big["bbox"])
            if inter < sa * 0.88:
                continue
            lines = [_normalized_block_text(x) for x in raw_big.splitlines() if x.strip()]
            if len(lines) < 2:
                continue
            if ts in lines:
                to_remove.add(small["id"])
                break

    images_sorted = sorted(image_blocks, key=lambda b: (_bbox_area(b["bbox"]), str(b["id"])))
    for i in range(len(images_sorted)):
        a = images_sorted[i]
        if a["id"] in to_remove:
            continue
        for j in range(i + 1, len(images_sorted)):
            b = images_sorted[j]
            if b["id"] in to_remove:
                continue
            if _bbox_iou(a["bbox"], b["bbox"]) < 0.86:
                continue
            to_remove.add(b["id"])

    return [b for b in page_blocks if b["id"] not in to_remove]


def _can_merge_inline_text_blocks(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ta = str(a.get("text", "")).strip()
    tb = str(b.get("text", "")).strip()
    if not ta or not tb:
        return False
    if _looks_like_toc_entry(ta) or _looks_like_toc_entry(tb):
        return False
    if _looks_like_toc_heading(ta) or _looks_like_toc_heading(tb):
        return False
    if "\n" in ta or "\n" in tb:
        return False
    if len(ta) + len(tb) > 420:
        return False

    ax0, ay0, ax1, ay1 = [float(v) for v in a["bbox"]]
    bx0, by0, bx1, by1 = [float(v) for v in b["bbox"]]
    ha = max(ay1 - ay0, 1.0)
    hb = max(by1 - by0, 1.0)
    if abs(ay0 - by0) > min(ha, hb) * 0.55 + 6.0:
        return False
    if abs(ay1 - by1) > min(ha, hb) * 0.55 + 8.0:
        return False
    if ha / hb > 2.2 or hb / ha > 2.2:
        return False

    gap = bx0 - ax1
    if gap > 36.0:
        return False
    if gap < -min(ha, hb) * 0.35:
        return False
    return True


def _combine_inline_chain(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    chain = sorted(chain, key=lambda b: (float(b["bbox"][0]), float(b["bbox"][1])))
    x0 = min(float(b["bbox"][0]) for b in chain)
    y0 = min(float(b["bbox"][1]) for b in chain)
    x1 = max(float(b["bbox"][2]) for b in chain)
    y1 = max(float(b["bbox"][3]) for b in chain)
    text = " ".join(_normalized_block_text(str(b.get("text", ""))) for b in chain if str(b.get("text", "")).strip())
    base = dict(chain[0])
    base["text"] = text
    base["bbox"] = _clamp_bbox([x0, y0, x1, y1], float(base["page_width"]), float(base["page_height"]))
    base["link_xref"] = next((b.get("link_xref") for b in chain if b.get("link_xref")), None)
    return base


def _merge_inline_fragments_page(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    images = [b for b in page_blocks if b.get("is_image")]
    texts = [b for b in page_blocks if not b.get("is_image")]
    if len(texts) < 2:
        return page_blocks

    texts.sort(key=lambda b: (float(b["bbox"][1]), float(b["bbox"][0])))
    merged_texts: List[Dict[str, Any]] = []
    i = 0
    while i < len(texts):
        chain = [texts[i]]
        j = i + 1
        while j < len(texts) and _can_merge_inline_text_blocks(chain[-1], texts[j]):
            chain.append(texts[j])
            j += 1
        if len(chain) == 1:
            merged_texts.append(chain[0])
        else:
            merged_texts.append(_combine_inline_chain(chain))
        i = j

    out = images + merged_texts
    out.sort(key=lambda b: (float(b["bbox"][1]), float(b["bbox"][0])))
    return out


def _assign_sequential_block_ids(blocks: List[Dict[str, Any]]) -> None:
    blocks.sort(key=lambda b: (int(b["page"]), float(b["bbox"][1]), float(b["bbox"][0])))
    for idx, block in enumerate(blocks):
        block["id"] = f"b{idx}"


def _finalize_extracted_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        by_page.setdefault(int(b["page"]), []).append(b)

    out: List[Dict[str, Any]] = []
    for page_no in sorted(by_page.keys()):
        pb = by_page[page_no]
        pb = _dedupe_page_blocks(pb)
        pb = _merge_inline_fragments_page(pb)
        out.extend(pb)
    _assign_sequential_block_ids(out)
    return out


def _to_real_bbox(bbox: List[float], page_width: float, page_height: float) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    page_width = max(float(page_width or 1.0), 1.0)
    page_height = max(float(page_height or 1.0), 1.0)

    # Convert normalized 0-1000 coordinates to real PDF coordinates when detected.
    if x1 > page_width + 1.0 or y1 > page_height + 1.0:
        x0 = x0 * page_width / 1000.0
        x1 = x1 * page_width / 1000.0
        y0 = y0 * page_height / 1000.0
        y1 = y1 * page_height / 1000.0

    return _clamp_bbox([x0, y0, x1, y1], page_width, page_height)


def _extract_toc_items(text: str) -> List[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines or [text.strip()] if text.strip() else []


def _split_toc_entry_parts(text: str) -> Dict[str, str]:
    content = (text or "").strip()
    if not content:
        return {}
    match = re.match(r"^(.*?)(\.{2,})\s*(\d+)\s*$", content)
    if not match:
        title_text = content
        page_no = ""
    else:
        title_text = match.group(1).strip()
        page_no = match.group(3).strip()

    number = ""
    title = title_text
    title_match = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", title_text)
    if title_match:
        number = title_match.group(1).strip()
        title = title_match.group(2).strip()

    result: Dict[str, str] = {}
    if number:
        result["number"] = number
    if title:
        result["title"] = title
    if page_no:
        result["page"] = page_no
    return result or {"title": content}


def _looks_like_toc_heading(text: str) -> bool:
    low = (text or "").strip().lower()
    return bool(low and re.search(r"\btable of contents\b", low))


def _looks_like_toc_entry(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or "\n" in stripped:
        return False
    return bool(_TOC_ENTRY_RE.match(stripped))


def _is_prose_paragraph_text(text: str) -> bool:
    """True for typical body copy — should never be forced to H1/H2 by heuristics."""
    content = (text or "").strip()
    if not content:
        return False
    if "\n" in content:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return True
    words = _split_words(content, max_words=200)
    wc = len(words)
    if wc >= 40:
        return True
    if len(content) >= 220 and wc >= 18:
        return True
    # Multiple sentences (not a one-line section title).
    sentence_breaks = len(re.findall(r"(?<=[.!?])\s+[A-Z(]", content))
    if sentence_breaks >= 2:
        return True
    if content.count(". ") >= 2 or content.count("? ") >= 2:
        return True
    return False


def _heading_tag_from_numbered_text(text: str) -> Optional[str]:
    stripped = (text or "").strip()
    if not stripped or _looks_like_toc_entry(stripped):
        return None
    if _is_prose_paragraph_text(stripped):
        return None
    # Section titles are short; long blocks that start with digits are still body text.
    if len(stripped) > 200 or len(_split_words(stripped, max_words=80)) > 28:
        return None
    match = re.match(r"^(\d+(?:\.\d+)*)\s+.+$", stripped)
    if not match:
        return None
    depth = match.group(1).count(".") + 1
    heading_level = min(depth + 1, 6)
    return f"H{heading_level}"


def _is_toc_container_text(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    entry_count = sum(1 for line in lines if _looks_like_toc_entry(line))
    return entry_count >= 2


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _line_bbox_from_words(
    line_text: str,
    container_bbox: List[float],
    page_words: List[Tuple[float, float, float, float, str]],
    page_width: float,
    page_height: float,
) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in container_bbox]
    words_in_box: List[Tuple[float, float, float, float, str]] = []
    for wx0, wy0, wx1, wy1, wtext in page_words:
        cx = (wx0 + wx1) / 2.0
        cy = (wy0 + wy1) / 2.0
        if x0 - 2.0 <= cx <= x1 + 2.0 and y0 - 2.0 <= cy <= y1 + 2.0:
            words_in_box.append((wx0, wy0, wx1, wy1, wtext))

    if not words_in_box:
        return _clamp_bbox(container_bbox, page_width, page_height)

    target = _normalize_text_for_match(line_text)
    line_groups: Dict[float, List[Tuple[float, float, float, float, str]]] = {}
    for word in words_in_box:
        line_groups.setdefault(round(word[1], 1), []).append(word)

    best_group: List[Tuple[float, float, float, float, str]] | None = None
    best_score = -1
    for group in line_groups.values():
        ordered = sorted(group, key=lambda item: (item[0], item[1]))
        joined = _normalize_text_for_match(" ".join(item[4] for item in ordered))
        score = 0
        if joined == target:
            score = 1000
        else:
            target_tokens = [token for token in re.split(r"\s+", target) if token]
            score = sum(1 for token in target_tokens if token in joined)
        if score > best_score:
            best_score = score
            best_group = ordered

    if not best_group or best_score <= 0:
        return _tighten_text_bbox(container_bbox, page_words, page_width, page_height)

    bx0 = min(item[0] for item in best_group) - 2.0
    by0 = min(item[1] for item in best_group) - 2.0
    bx1 = max(item[2] for item in best_group) + 2.0
    by1 = max(item[3] for item in best_group) + 2.0
    return _clamp_bbox([bx0, by0, bx1, by1], page_width, page_height)


def _should_split_multiline_block(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2 or len(lines) > 4:
        return False
    if any(len(line) > 90 for line in lines):
        return False
    if _is_toc_container_text(text):
        return False
    joined = " ".join(lines)
    if len(joined) > 220:
        return False
    return True


def _explode_multiline_block(
    block_id_start: int,
    page_no: int,
    text: str,
    bbox: List[float],
    page_words: List[Tuple[float, float, float, float, str]],
    page_width: float,
    page_height: float,
) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    results: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line_bbox = _line_bbox_from_words(line, bbox, page_words, page_width, page_height)
        results.append(
            {
                "id": f"b{block_id_start + idx}",
                "page": page_no,
                "text": line,
                "bbox": line_bbox,
                "is_image": False,
                "image_xref": None,
                "image_order": None,
                "image_width": None,
                "image_height": None,
                "link_xref": None,
                "bbox_normalized": False,
                "page_width": page_width,
                "page_height": page_height,
            }
        )
    return results


def _explode_toc_block(
    block_id_start: int,
    page_no: int,
    text: str,
    bbox: List[float],
    page_words: List[Tuple[float, float, float, float, str]],
    page_width: float,
    page_height: float,
) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []

    results: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line_bbox = _line_bbox_from_words(line, bbox, page_words, page_width, page_height)
        results.append(
            {
                "id": f"b{block_id_start + idx}",
                "page": page_no,
                "text": line,
                "bbox": line_bbox,
                "is_image": False,
                "image_xref": None,
                "image_order": None,
                "image_width": None,
                "image_height": None,
                "bbox_normalized": False,
                "page_width": page_width,
                "page_height": page_height,
            }
        )
    return results


def _extract_list_items(text: str) -> List[str]:
    items: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^(?:[-*\u2022]+|\d+[.)])\s*", "", stripped)
        items.append(stripped)
    if not items and text.strip():
        items = [text.strip()]
    return items


def _extract_table_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped) if cell.strip()]
        if not cells:
            cells = [stripped]
        rows.append(cells)
    if not rows and text.strip():
        rows = [[text.strip()]]
    return rows


def _heuristic_tag(text: str, fallback: str = "Paragraph") -> str:
    content = (text or "").strip()
    low = content.lower()
    if not content:
        return fallback
    if low.startswith("table"):
        return "Table"
    if low.startswith("reference") or low.startswith("references"):
        return "Reference"
    if re.search(r"\b(contents|table of contents)\b", low):
        return "TOC"
    if re.match(r"^(?:[-*\u2022]|\d+[.)])\s+", content):
        return "List"
    if len(content) < 70 and content.isupper():
        return "H1"
    return fallback


def _is_list_like_text(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    return bool(re.match(r"^(?:[-*\u2022]|\d+[.)]|[A-Za-z][.)])\s+", content))


def _is_list_compatible_block(block: Dict[str, Any]) -> bool:
    text = (block.get("text", "") or "").strip()
    label = str(block.get("predicted_label", ""))
    if not text or label in {"Figure", "Table", "TOC", "Reference", "Header", "Footer", "Caption"}:
        return False
    if _looks_like_toc_heading(text) or _looks_like_toc_entry(text):
        return False
    return True


def _promote_list_neighbors(predicted_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(predicted_blocks, key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))
    promoted = [dict(block) for block in ordered]

    for idx, block in enumerate(promoted):
        if str(block.get("predicted_label")) != "Paragraph":
            continue
        if not _is_list_compatible_block(block):
            continue
        if _is_list_like_text(block.get("text", "")):
            block["predicted_label"] = "List"
            continue

        page = int(block["page"])
        x0, y0, x1, y1 = [float(v) for v in block["bbox"]]
        height = max(1.0, y1 - y0)

        matching_list_neighbors = 0
        for neighbor_idx in (idx - 1, idx + 1):
            if neighbor_idx < 0 or neighbor_idx >= len(promoted):
                continue
            neighbor = promoted[neighbor_idx]
            if int(neighbor["page"]) != page:
                continue
            if str(neighbor.get("predicted_label")) != "List":
                continue
            if not _is_list_compatible_block(neighbor):
                continue

            nx0, ny0, nx1, ny1 = [float(v) for v in neighbor["bbox"]]
            nheight = max(1.0, ny1 - ny0)
            same_indent = abs(x0 - nx0) <= max(12.0, min(height, nheight) * 0.8)
            vertical_gap = max(0.0, max(y0, ny0) - min(y1, ny1))
            close_enough = vertical_gap <= max(18.0, max(height, nheight) * 1.4)
            similar_width = abs((x1 - x0) - (nx1 - nx0)) <= max(24.0, min(x1 - x0, nx1 - nx0) * 0.35)

            if same_indent and close_enough and similar_width:
                matching_list_neighbors += 1

        # Only upgrade contextually when the block is surrounded by list items.
        # This avoids pulling normal paragraph lead-in text into the list.
        if matching_list_neighbors >= 2:
            block["predicted_label"] = "List"

    return promoted


def _is_likely_author_or_affiliation(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    low = content.lower()
    affiliation_terms = (
        "school",
        "university",
        "business",
        "law",
        "editor",
        "published",
        "series",
    )
    if any(term in low for term in affiliation_terms):
        return True
    words = [word for word in re.split(r"\s+", content) if word]
    if 1 <= len(words) <= 5 and content.upper() == content:
        return True
    return False


def _is_running_header_footer_text(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    low = content.lower()
    if "copyright" in low or "all rights reserved" in low:
        return True
    if "|" in content and len(content) <= 80:
        return True
    if re.match(r"^\d+\s*(?:[|:-]\s*.+)?$", content):
        return True
    if re.match(r"^.+\s+[|:-]\s+\d+$", content):
        return True
    return False


def _refine_heading_labels(predicted_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refined = [dict(block) for block in predicted_blocks]
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for block in refined:
        by_page.setdefault(int(block["page"]), []).append(block)

    for page_blocks in by_page.values():
        text_blocks = [
            block
            for block in page_blocks
            if not block.get("is_image") and str(block.get("text", "")).strip()
        ]
        if not text_blocks:
            continue

        heights = [max(float(b["bbox"][3]) - float(b["bbox"][1]), 1.0) for b in text_blocks]
        max_height = max(heights)

        best_candidate: Dict[str, Any] | None = None
        best_score = -1.0

        for block in text_blocks:
            text = str(block.get("text", "")).strip()
            label = str(block.get("predicted_label", "Paragraph"))
            x0, y0, x1, y1 = [float(v) for v in block["bbox"]]
            width = max(x1 - x0, 1.0)
            height = max(y1 - y0, 1.0)
            area = width * height
            page_width = max(float(block.get("page_width", 1.0)), 1.0)
            page_height = max(float(block.get("page_height", 1.0)), 1.0)
            top_bias = max(0.0, 1.0 - (y0 / page_height))
            is_authorish = _is_likely_author_or_affiliation(text)
            is_running_text = _is_running_header_footer_text(text)
            word_count = len(_split_words(text, max_words=80))
            looks_sentence_like = any(ch.islower() for ch in text) and len(text) >= 10
            is_prose = _is_prose_paragraph_text(text)

            if label == "Figure" or _looks_like_toc_heading(text) or _looks_like_toc_entry(text):
                continue
            if is_running_text:
                continue
            if is_authorish and height < max_height * 0.9:
                continue
            # Never pick body paragraphs as the page "title" candidate.
            if is_prose or word_count > 22:
                continue

            score = 0.0
            score += min(height / max(max_height, 1.0), 1.5) * 5.0
            score += min(width / page_width, 1.0) * 2.0
            score += min(area / max(page_width * 40.0, 1.0), 2.0)
            score += top_bias * 1.5
            # Prose-like rhythm means body text, not a display heading.
            if looks_sentence_like:
                score -= 5.0
            if label in {"H1", "H2", "H3"}:
                score += 1.25
            if word_count > 12:
                score -= 3.0
            if word_count <= 14 and text.isupper():
                score += 2.5
            if is_authorish:
                score -= 3.5
            if is_running_text:
                score -= 4.0

            if score > best_score:
                best_score = score
                best_candidate = block

        if best_candidate is None:
            continue

        candidate_height = max(float(best_candidate["bbox"][3]) - float(best_candidate["bbox"][1]), 1.0)
        candidate_text = str(best_candidate.get("text", "")).strip()

        for block in text_blocks:
            text = str(block.get("text", "")).strip()
            label = str(block.get("predicted_label", "Paragraph"))
            height = max(float(block["bbox"][3]) - float(block["bbox"][1]), 1.0)
            numbered_heading_tag = _heading_tag_from_numbered_text(text)
            is_running_text = _is_running_header_footer_text(text)

            if is_running_text:
                block["predicted_label"] = "Paragraph"
                continue

            if numbered_heading_tag is not None:
                block["predicted_label"] = numbered_heading_tag
                continue

            if _is_prose_paragraph_text(text):
                block["predicted_label"] = "Paragraph"
                continue

            if block["id"] == best_candidate["id"]:
                block["predicted_label"] = "H1"
                continue

            if label == "H1":
                if (
                    _is_likely_author_or_affiliation(text)
                    or height < candidate_height * 0.72
                    or _is_prose_paragraph_text(text)
                ):
                    block["predicted_label"] = "Paragraph"
            elif label == "Paragraph":
                # Do not promote tall body blocks to H2; only short, title-like lines.
                wc = len(_split_words(text, max_words=80))
                if (
                    text != candidate_text
                    and not _is_likely_author_or_affiliation(text)
                    and wc <= 24
                    and len(text) <= 200
                    and not _is_prose_paragraph_text(text)
                    and height >= candidate_height * 0.72
                    and height >= max_height * 0.72
                ):
                    block["predicted_label"] = "H2"

        for block in text_blocks:
            pl = str(block.get("predicted_label", "Paragraph"))
            if pl not in {"H1", "H2", "H3", "H4", "H5", "H6"}:
                continue
            t = str(block.get("text", "")).strip()
            if _heading_tag_from_numbered_text(t) is not None:
                continue
            if _is_prose_paragraph_text(t):
                block["predicted_label"] = "Paragraph"
            elif pl == "H1" and len(_split_words(t, max_words=100)) > 18:
                block["predicted_label"] = "Paragraph"

    return refined


@lru_cache(maxsize=2)
def load_model(model_dir: str = "./model") -> Dict[str, Any]:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config_json = json.load(file)

    id2label = {int(k): v for k, v in config_json.get("id2label", {}).items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = LayoutLMv3Processor.from_pretrained(str(model_path), local_files_only=True)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "id2label": id2label,
    }


def extract_blocks(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, float]], int]:
    blocks: List[Dict[str, Any]] = []
    page_sizes: Dict[int, Dict[str, float]] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        block_id = 0
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_no = page_idx + 1
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            page_words: List[Tuple[float, float, float, float, str]] = []

            for word in page.get_text("words"):
                wx0, wy0, wx1, wy1, wtext = word[:5]
                if not str(wtext or "").strip():
                    continue
                page_words.append((float(wx0), float(wy0), float(wx1), float(wy1), str(wtext)))

            page_sizes[page_no] = {
                "width": page_width,
                "height": page_height,
            }

            page_link_annots: List[Dict[str, Any]] = []
            for link in page.get_links():
                link_xref = int(link.get("xref") or 0)
                link_rect = link.get("from")
                if link_xref <= 0 or link_rect is None:
                    continue
                page_link_annots.append(
                    {
                        "xref": link_xref,
                        "bbox": _clamp_bbox(
                            [float(link_rect.x0), float(link_rect.y0), float(link_rect.x1), float(link_rect.y1)],
                            page_width,
                            page_height,
                        ),
                    }
                )

            page_seen_boxes = set()

            # Explicit image extraction via xref is more reliable than block-type alone.
            for image_order, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                image_width = int(image_info[2]) if len(image_info) > 2 else None
                image_height = int(image_info[3]) if len(image_info) > 3 else None
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []

                for rect in rects:
                    bbox = _clamp_bbox(
                        [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                        page_width,
                        page_height,
                    )
                    key = _bbox_key(bbox)
                    if key in page_seen_boxes:
                        continue
                    page_seen_boxes.add(key)
                    blocks.append(
                        {
                            "id": f"b{block_id}",
                            "page": page_no,
                            "text": "",
                            "bbox": bbox,
                            "is_image": True,
                            "image_xref": int(xref),
                            "image_order": int(image_order),
                            "image_width": image_width,
                            "image_height": image_height,
                            "image_source": "pymupdf_image_xref",
                            "link_xref": None,
                            "bbox_normalized": False,
                            "page_width": page_width,
                            "page_height": page_height,
                        }
                    )
                    block_id += 1

            for raw in page.get_text("blocks"):
                x0, y0, x1, y1, text, _, block_type = raw[:7]
                text = (text or "").strip()
                is_image = int(block_type) == 1

                if not is_image and not text:
                    continue

                bbox = _clamp_bbox([float(x0), float(y0), float(x1), float(y1)], page_width, page_height)
                
                # Only apply light tightening to small text blocks; skip large blocks (images/containers).
                if not is_image and text and len(text) > 5:
                    tightened = _tighten_text_bbox(bbox, page_words, page_width, page_height)
                    # Prefer the word-based bounds unless they collapse to an
                    # implausibly tiny box; this keeps long TOC lines from
                    # stretching far past the final page number.
                    orig_width = max(bbox[2] - bbox[0], 1.0)
                    orig_height = max(bbox[3] - bbox[1], 1.0)
                    tight_width = max(tightened[2] - tightened[0], 1.0)
                    tight_height = max(tightened[3] - tightened[1], 1.0)
                    if tight_width > orig_width * 0.1 and tight_height > orig_height * 0.2:
                        bbox = tightened

                if not is_image and _is_toc_container_text(text):
                    exploded = _explode_toc_block(
                        block_id,
                        page_no,
                        text,
                        bbox,
                        page_words,
                        page_width,
                        page_height,
                    )
                    if exploded:
                        for item in exploded:
                            item["link_xref"] = None
                            best_overlap = 0.0
                            for link_annot in page_link_annots:
                                overlap = _bbox_intersection_area(item["bbox"], link_annot["bbox"])
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    item["link_xref"] = int(link_annot["xref"])
                        blocks.extend(exploded)
                        block_id += len(exploded)
                        continue

                if not is_image and _should_split_multiline_block(text):
                    exploded = _explode_multiline_block(
                        block_id,
                        page_no,
                        text,
                        bbox,
                        page_words,
                        page_width,
                        page_height,
                    )
                    if exploded:
                        blocks.extend(exploded)
                        block_id += len(exploded)
                        continue
                
                key = _bbox_key(bbox)

                # Avoid duplicate image regions when both extractors detect the same area.
                if is_image and key in page_seen_boxes:
                    continue

                if is_image:
                    page_seen_boxes.add(key)

                blocks.append(
                    {
                        "id": f"b{block_id}",
                        "page": page_no,
                        "text": text,
                        "bbox": bbox,
                        "is_image": is_image,
                        "image_xref": int(raw[7]) if is_image and len(raw) > 7 and str(raw[7]).isdigit() else None,
                        "image_order": None,
                        "image_width": None,
                        "image_height": None,
                        "link_xref": None,
                        "bbox_normalized": False,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                )
                block_id += 1

        blocks = _finalize_extracted_blocks(blocks)
        return blocks, page_sizes, doc.page_count


def predict_blocks(
    pdf_bytes: bytes,
    blocks: List[Dict[str, Any]],
    model_dir: str = "./model",
    batch_size: int = 12,
) -> List[Dict[str, Any]]:
    bundle = load_model(model_dir)
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]
    id2label = bundle["id2label"]

    results: Dict[str, Dict[str, Any]] = {}
    batch_items: List[Dict[str, Any]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_images: Dict[int, Image.Image] = {}

        def get_page_image(page_no: int) -> Image.Image:
            if page_no not in page_images:
                page = doc[page_no - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                page_images[page_no] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return page_images[page_no]

        for block in blocks:
            if block["is_image"]:
                results[block["id"]] = {
                    **block,
                    "predicted_label": "Figure",
                    "confidence": 0.99,
                }
                continue

            text = block.get("text", "").strip()
            if not text:
                results[block["id"]] = {
                    **block,
                    "predicted_label": "Figure",
                    "confidence": 0.75,
                }
                continue

            words = _split_words(text)
            norm_box = _normalize_bbox(block["bbox"], block["page_width"], block["page_height"])
            word_boxes = [norm_box for _ in words]

            page_image = get_page_image(block["page"])
            sx = page_image.width / max(block["page_width"], 1.0)
            sy = page_image.height / max(block["page_height"], 1.0)
            x0, y0, x1, y1 = block["bbox"]
            crop_box = (
                int(max(0, x0 * sx)),
                int(max(0, y0 * sy)),
                int(min(page_image.width, x1 * sx)),
                int(min(page_image.height, y1 * sy)),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                crop = page_image
            else:
                crop = page_image.crop(crop_box)

            batch_items.append(
                {
                    "block": block,
                    "image": crop,
                    "words": words,
                    "boxes": word_boxes,
                }
            )

        for i in range(0, len(batch_items), batch_size):
            chunk = batch_items[i : i + batch_size]
            try:
                encoding = processor(
                    images=[item["image"] for item in chunk],
                    text=[item["words"] for item in chunk],
                    boxes=[item["boxes"] for item in chunk],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

                tensor_inputs = {k: v.to(device) for k, v in encoding.items() if hasattr(v, "to")}
                with torch.no_grad():
                    logits = model(**tensor_inputs).logits
                    probs = torch.softmax(logits, dim=-1)

                labels = torch.argmax(probs, dim=-1).tolist()
                confs = torch.max(probs, dim=-1).values.tolist()

                for item, label_idx, confidence in zip(chunk, labels, confs):
                    block = item["block"]
                    predicted = id2label.get(int(label_idx), "Paragraph")
                    results[block["id"]] = {
                        **block,
                        "predicted_label": predicted,
                        "confidence": float(confidence),
                    }
            except Exception:
                # If model inference fails for a chunk, preserve flow with a heuristic fallback.
                for item in chunk:
                    block = item["block"]
                    heuristic = _heuristic_tag(block.get("text", ""))
                    results[block["id"]] = {
                        **block,
                        "predicted_label": heuristic,
                        "confidence": 0.5,
                    }

    ordered = [results[block["id"]] for block in blocks if block["id"] in results]
    ordered = _promote_list_neighbors(ordered)
    return _refine_heading_labels(ordered)


def _make_leaf_node(block: Dict[str, Any], tag: str) -> Dict[str, Any]:
    text = block.get("text", "")
    if tag == "Figure" and not str(text or "").strip():
        image_xref = block.get("image_xref")
        image_width = block.get("image_width")
        image_height = block.get("image_height")
        details: List[str] = []
        if image_width:
            details.append(f"w:{int(image_width)}")
        if image_height:
            details.append(f"h:{int(image_height)}")
        suffix = f": {' '.join(details)}" if details else ""
        if image_xref is not None:
            text = f"image ({int(image_xref)}){suffix}"
        else:
            text = f"image{suffix}" if suffix else "image"

    return {
        "id": f"n-{block['id']}",
        "tag": tag,
        "text": text,
        "page": block["page"],
        "bbox": block["bbox"],
        "coord_page_width": float(block.get("coord_page_width") or block.get("page_width", 1.0)),
        "coord_page_height": float(block.get("coord_page_height") or block.get("page_height", 1.0)),
        "image_xref": block.get("image_xref"),
        "image_order": block.get("image_order"),
        "image_width": block.get("image_width"),
        "image_height": block.get("image_height"),
        "link_xref": block.get("link_xref"),
        "block_ids": [block["id"]],
        "children": [],
    }


def _aggregate_block_ids(node: Dict[str, Any]) -> List[str]:
    ids = list(node.get("block_ids", []))
    for child in node.get("children", []):
        ids.extend(_aggregate_block_ids(child))
    # Preserve order while removing duplicates.
    seen = set()
    unique: List[str] = []
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    node["block_ids"] = unique
    return unique


def group_blocks(predicted_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    pages: Dict[int, Dict[str, Any]] = {}
    ordered_flat_blocks: List[Dict[str, Any]] = []

    tree: Dict[str, Any] = {
        "id": "document",
        "tag": "Document",
        "text": "",
        "page": None,
        "bbox": None,
        "block_ids": [],
        "children": [],
    }

    first_h1_seen = False

    for block in predicted_blocks:
        page_no = int(block["page"])
        pages.setdefault(page_no, {"page": page_no, "blocks": []})

        predicted = block.get("predicted_label", "Paragraph")
        final_tag = predicted
        text = block.get("text", "")
        numbered_heading_tag = _heading_tag_from_numbered_text(text)

        if _looks_like_toc_heading(text):
            final_tag = "TOC"
        elif _looks_like_toc_entry(text):
            final_tag = "TOCI"
        elif numbered_heading_tag is not None:
            final_tag = numbered_heading_tag

        if not first_h1_seen:
            if predicted == "H1":
                first_h1_seen = True
            elif final_tag in {"TOC", "TOCI"}:
                pass
            else:
                final_tag = "Figure" if (predicted == "Figure" or block.get("is_image")) else "Paragraph"

        if final_tag == "H1":
            first_h1_seen = True

        # Normalize model labels to output tag schema.
        if final_tag == "Paragraph":
            final_tag = "P"
        elif final_tag == "List":
            final_tag = "L"

        flat_block = {
            "id": block["id"],
            "tag": final_tag,
            "text": block.get("text", ""),
            "bbox": _to_real_bbox(
                block["bbox"],
                float(block.get("page_width", 1.0)),
                float(block.get("page_height", 1.0)),
            ),
            "page": page_no,
            "coord_page_width": float(block.get("page_width", 1.0)),
            "coord_page_height": float(block.get("page_height", 1.0)),
            "image_xref": block.get("image_xref"),
            "image_order": block.get("image_order"),
            "image_width": block.get("image_width"),
            "image_height": block.get("image_height"),
            "link_xref": block.get("link_xref"),
            "confidence": float(block.get("confidence", 0.0)),
        }
        pages[page_no]["blocks"].append(flat_block)

    # Keep per-page buckets for fast rendering, but also maintain a global reading-order stream.
    for page_no in sorted(pages.keys()):
        pages[page_no]["blocks"].sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        ordered_flat_blocks.extend(pages[page_no]["blocks"])

    ordered_flat_blocks.sort(key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))

    current_toc_node: Dict[str, Any] | None = None

    for block in ordered_flat_blocks:
        tag = block["tag"]
        text = block.get("text", "")
        page_no = int(block["page"])

        if current_toc_node is not None and int(current_toc_node.get("page") or 0) != page_no:
            current_toc_node = None

        if tag == "TOC":
            toc_node = {
                **_make_leaf_node(block, "TOC"),
                "children": [],
            }
            tree["children"].append(toc_node)
            current_toc_node = toc_node
        elif tag == "TOCI":
            toc_parts = _split_toc_entry_parts(text)
            number_text = toc_parts.get("number", "").strip()
            title_text = toc_parts.get("title", "").strip()
            page_text = toc_parts.get("page", "").strip()
            link_text_parts = [part for part in [number_text, title_text, page_text] if part]
            link_text = " ".join(link_text_parts).strip() or text
            link_node = {
                "id": f"n-{block['id']}-link",
                "tag": "Link",
                "text": link_text,
                "page": block["page"],
                "bbox": block["bbox"],
                "coord_page_width": float(block.get("coord_page_width", 1.0)),
                "coord_page_height": float(block.get("coord_page_height", 1.0)),
                "link_xref": block.get("link_xref"),
                "link_text_parts": link_text_parts,
                "block_ids": [block["id"]],
                "children": [],
            }
            toci_node = {
                **_make_leaf_node(block, "TOCI"),
                "children": [link_node],
            }
            if current_toc_node is not None:
                current_toc_node["children"].append(toci_node)
            else:
                tree["children"].append(toci_node)
        elif tag == "Table":
            current_toc_node = None
            table_node = {
                **_make_leaf_node(block, "Table"),
                "children": [],
            }
            for row_idx, row in enumerate(_extract_table_rows(text), start=1):
                tr_node = {
                    "id": f"{table_node['id']}-tr-{row_idx}",
                    "tag": "TR",
                    "text": "",
                    "page": block["page"],
                    "bbox": block["bbox"],
                    "coord_page_width": float(block.get("coord_page_width", 1.0)),
                    "coord_page_height": float(block.get("coord_page_height", 1.0)),
                    "block_ids": [block["id"]],
                    "children": [],
                }
                for col_idx, cell in enumerate(row, start=1):
                    tr_node["children"].append(
                        {
                            "id": f"{tr_node['id']}-td-{col_idx}",
                            "tag": "TD",
                            "text": cell,
                            "page": block["page"],
                            "bbox": block["bbox"],
                            "coord_page_width": float(block.get("coord_page_width", 1.0)),
                            "coord_page_height": float(block.get("coord_page_height", 1.0)),
                            "block_ids": [block["id"]],
                            "children": [],
                        }
                    )
                table_node["children"].append(tr_node)
            tree["children"].append(table_node)
        elif tag == "L":
            current_toc_node = None
            list_node = {
                **_make_leaf_node(block, "L"),
                "children": [],
            }
            for idx, item in enumerate(_extract_list_items(text), start=1):
                list_node["children"].append(
                    {
                        "id": f"{list_node['id']}-li-{idx}",
                        "tag": "LI",
                        "text": item,
                        "page": block["page"],
                        "bbox": block["bbox"],
                        "coord_page_width": float(block.get("coord_page_width", 1.0)),
                        "coord_page_height": float(block.get("coord_page_height", 1.0)),
                        "block_ids": [block["id"]],
                        "children": [],
                    }
                )
            tree["children"].append(list_node)
        elif tag == "Reference":
            current_toc_node = None
            tree["children"].append(_make_leaf_node(block, "Reference"))
        else:
            if tag in {"H1", "H2", "Figure"}:
                current_toc_node = None
            tree["children"].append(_make_leaf_node(block, tag))

    _aggregate_block_ids(tree)

    return {
        "pages": [pages[k] for k in sorted(pages.keys())],
        "blocks": ordered_flat_blocks,
        "tree": tree,
    }


def make_doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def render_page_png(pdf_bytes: bytes, page_number: int, zoom: float = 1.8) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number < 1 or page_number > doc.page_count:
            raise IndexError("page_number out of range")
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _pdf_structure_tag(tag: str) -> str:
    normalized = (tag or "P").strip()
    role_map = {
        "Document": "Document",
        "H1": "H1",
        "H2": "H2",
        "H3": "H3",
        "H4": "H4",
        "H5": "H5",
        "H6": "H6",
        "P": "P",
        "Span": "Span",
        "Link": "Link",
        "TOC": "TOC",
        "TOCI": "TOCI",
        "Table": "Table",
        "TR": "TR",
        "TD": "TD",
        "L": "L",
        "LI": "LI",
        "Figure": "Figure",
        "Reference": "BibEntry",
    }
    return role_map.get(normalized, "P")


def _assign_mcids(node: Dict[str, Any], next_mcid_by_page: Dict[int, int]) -> None:
    children = node.get("children", []) or []
    link_text_parts = node.get("link_text_parts") or []
    text = (node.get("text") or "").strip()
    page_no = int(node.get("page") or 0)
    tag = str(node.get("tag") or "").strip()
    has_bbox = bool(node.get("bbox"))

    if children:
        node.pop("_mcid", None)
        for child in children:
            _assign_mcids(child, next_mcid_by_page)
    elif not link_text_parts and page_no > 0 and (text or has_bbox or tag == "Figure"):
        node["_mcid"] = next_mcid_by_page.get(page_no, 0)
        next_mcid_by_page[page_no] = node["_mcid"] + 1
    else:
        node.pop("_mcid", None)

    if link_text_parts and page_no > 0:
        part_mcids: List[Dict[str, Any]] = []
        for part_text in link_text_parts:
            part_mcid = next_mcid_by_page.get(page_no, 0)
            next_mcid_by_page[page_no] = part_mcid + 1
            part_mcids.append({"text": str(part_text), "mcid": part_mcid})
        node["_link_part_mcids"] = part_mcids
    else:
        node.pop("_link_part_mcids", None)


def _pdf_literal_text(text: str) -> str:
    value = (text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({value})"


def _iter_marked_content_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for child in node.get("children", []) or []:
        items.extend(_iter_marked_content_nodes(child))
    if "_mcid" in node:
        items.append(node)
    for part in node.get("_link_part_mcids", []) or []:
        items.append(
            {
                "id": f"{node.get('id', 'node')}-part-{part['mcid']}",
                "tag": str(node.get("tag") or "Link"),
                "text": str(part.get("text") or ""),
                "page": node.get("page"),
                "bbox": node.get("bbox"),
                "coord_page_width": node.get("coord_page_width"),
                "coord_page_height": node.get("coord_page_height"),
                "_mcid": int(part["mcid"]),
            }
        )
    return items


def _ensure_resource_dict(page: Any, key: str) -> DictionaryObject:
    if key in page:
        resource = page[NameObject(key)]
        if hasattr(resource, "get_object"):
            resource = resource.get_object()
        if isinstance(resource, DictionaryObject):
            return resource
    resource = DictionaryObject()
    page[NameObject(key)] = resource
    return resource


def _ensure_overlay_resources(writer: PdfWriter, page: Any) -> Tuple[str, str]:
    resources = _ensure_resource_dict(page, "/Resources")
    font_dict = _ensure_resource_dict(resources, "/Font")
    ext_gstate_dict = _ensure_resource_dict(resources, "/ExtGState")

    font_name = "/FTag"
    gs_name = "/GSTag"

    if NameObject(font_name) not in font_dict:
        font_ref = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                    NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
                }
            )
        )
        font_dict[NameObject(font_name)] = font_ref

    if NameObject(gs_name) not in ext_gstate_dict:
        gs_ref = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/ExtGState"),
                    # Keep overlay geometry available to Acrobat while making it
                    # effectively invisible to the eye.
                    NameObject("/ca"): FloatObject(0.001),
                    NameObject("/CA"): FloatObject(0.001),
                }
            )
        )
        ext_gstate_dict[NameObject(gs_name)] = gs_ref

    return font_name, gs_name


def _build_page_xobject_map(page: Any) -> Dict[str, Any]:
    xobject_map: Dict[str, Any] = {}
    resources = page.get("/Resources")
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    if not isinstance(resources, DictionaryObject):
        return xobject_map

    xobjects = resources.get("/XObject")
    if hasattr(xobjects, "get_object"):
        xobjects = xobjects.get_object()
    if not isinstance(xobjects, DictionaryObject):
        return xobject_map

    ordered_refs: List[Any] = []
    size_map: Dict[Tuple[int, int], List[Any]] = {}
    for name in xobjects.keys():
        ref = xobjects.raw_get(name) if hasattr(xobjects, "raw_get") else xobjects[name]
        ordered_refs.append(ref)
        obj = ref.get_object() if hasattr(ref, "get_object") else ref
        width = int(obj.get("/Width", 0) or 0)
        height = int(obj.get("/Height", 0) or 0)
        size_map.setdefault((width, height), []).append(ref)
        if hasattr(ref, "idnum"):
            xobject_map[f"id:{int(ref.idnum)}"] = ref
    xobject_map["ordered"] = ordered_refs
    xobject_map["by_size"] = size_map
    return xobject_map


def _build_page_annotation_map(page: Any) -> Dict[str, Any]:
    annotation_map: Dict[str, Any] = {}
    annots = page.get("/Annots")
    if hasattr(annots, "get_object"):
        annots = annots.get_object()
    if not isinstance(annots, ArrayObject):
        return annotation_map

    for annot_ref in annots:
        ref = annot_ref if hasattr(annot_ref, "get_object") else None
        if ref is not None and hasattr(ref, "idnum"):
            annotation_map[f"id:{int(ref.idnum)}"] = ref
    return annotation_map


def _resolve_figure_xobject_ref(node: Dict[str, Any], page_xobject_map: Dict[str, Any]) -> Any:
    image_xref = node.get("image_xref")
    if image_xref is not None:
        ref = page_xobject_map.get(f"id:{int(image_xref)}")
        if ref is not None:
            return ref

    image_order = node.get("image_order")
    ordered_refs = page_xobject_map.get("ordered") or []
    if image_order is not None and 0 <= int(image_order) < len(ordered_refs):
        return ordered_refs[int(image_order)]

    image_width = int(node.get("image_width") or 0)
    image_height = int(node.get("image_height") or 0)
    if image_width > 0 and image_height > 0:
        refs = page_xobject_map.get("by_size", {}).get((image_width, image_height)) or []
        if len(refs) == 1:
            return refs[0]

    return None


def _flatten_content_entries(content_obj: Any, seen: set[int] | None = None) -> ArrayObject:
    flattened = ArrayObject()
    seen = seen or set()

    ref_key: int | None = None
    if hasattr(content_obj, "idnum") and hasattr(content_obj, "generation"):
        ref_key = (int(content_obj.idnum) << 16) + int(content_obj.generation)
        if ref_key in seen:
            return flattened
        seen.add(ref_key)

    resolved = content_obj.get_object() if hasattr(content_obj, "get_object") else content_obj

    if isinstance(resolved, ArrayObject):
        for item in resolved:
            nested = _flatten_content_entries(item, seen)
            for nested_item in nested:
                flattened.append(nested_item)
    elif content_obj is not None:
        flattened.append(content_obj)

    if ref_key is not None:
        seen.discard(ref_key)

    return flattened


def _append_overlay_stream(writer: PdfWriter, page: Any, content: bytes) -> None:
    stream = DecodedStreamObject()
    stream.set_data(content)
    stream_ref = writer._add_object(stream)

    existing = page.raw_get("/Contents") if "/Contents" in page else None
    if existing is None:
        page[NameObject("/Contents")] = stream_ref
        return

    contents = _flatten_content_entries(existing)
    contents.append(stream_ref)
    page[NameObject("/Contents")] = contents


def _build_page_overlay_stream(
    page: Any,
    nodes: List[Dict[str, Any]],
    font_name: str,
    gs_name: str,
) -> bytes:
    if not nodes:
        return b""

    page_left = 0.0
    page_bottom = 0.0
    coord_w = 0.0
    coord_h = 0.0
    try:
        crop = page.cropbox
        page_left = float(crop.left)
        page_bottom = float(crop.bottom)
        coord_w = float(crop.width)
        coord_h = float(crop.height)
    except Exception:
        mediabox = page.mediabox
        page_left = float(mediabox.left)
        page_bottom = float(mediabox.bottom)
        coord_w = float(mediabox.width)
        coord_h = float(mediabox.height)

    for node in nodes:
        cw = float(node.get("coord_page_width") or 0.0)
        ch = float(node.get("coord_page_height") or 0.0)
        if cw > 0 and ch > 0:
            coord_w = cw
            coord_h = ch
            break

    page_height = coord_h
    parts: List[str] = []
    overlay_padding_x = 2.0
    overlay_padding_y = 1.0

    for node in sorted(nodes, key=lambda item: int(item["_mcid"])):
        bbox = node.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        tag = str(node.get("tag") or "").strip()
        page_width = coord_w
        x0 = max(0.0, x0 - overlay_padding_x)
        x1 = min(page_width, x1 + overlay_padding_x)
        y0 = max(0.0, y0 - overlay_padding_y)
        y1 = min(page_height, y1 + overlay_padding_y)
        box_width = max(x1 - x0, 1.0)
        box_height = max(y1 - y0, 1.0)
        lines = [line.strip() for line in (node.get("text") or "").splitlines() if line.strip()]
        if not lines:
            lines = [(node.get("text") or "").strip()]
        lines = [line for line in lines if line]

        # Keep other non-text nodes navigable by emitting a minimal anchor run.
        if not lines:
            lines = [" "] if tag in {"Figure", "Table"} else ["\u200b"]

        # Emit text-only marked content so Acrobat shows readable tag names
        # and a single highlight region instead of separate Path + text boxes.
        overlay_text = " ".join(line for line in lines if line).strip()
        if not overlay_text:
            overlay_text = "image" if tag == "Figure" else " "

        # Stretch a single invisible text run to the node bbox.
        font_size = max(min(box_height * 0.78, 72.0), 6.0)
        approx_char_width = max(font_size * 0.55, 1.0)
        estimated_text_width = max(len(overlay_text) * approx_char_width, 1.0)
        # Add extra width buffer so long labels do not get clipped at the box edges.
        horizontal_scale = max(5.0, min(1000.0, ((box_width * 1.05) / estimated_text_width) * 100.0))
        text_x = page_left + x0
        text_y = page_bottom + (page_height - y1) + max((box_height - font_size) * 0.55, 0.0)

        parts.append("q")
        parts.append(f"{gs_name} gs")
        parts.append(f"/{_pdf_structure_tag(tag)} <</MCID {int(node['_mcid'])}>> BDC")
        parts.append("BT")
        parts.append(f"{font_name} {font_size:.2f} Tf")
        parts.append(f"{horizontal_scale:.2f} Tz")
        parts.append("0 Tr")
        parts.append("0 g")
        parts.append(f"1 0 0 1 {text_x:.2f} {text_y:.2f} Tm")
        parts.append(f"{_pdf_literal_text(overlay_text)} Tj")
        parts.append("ET")
        parts.append("EMC")
        parts.append("Q")

    return ("\n".join(parts) + "\n").encode("utf-8")


def _append_structure_node(
    writer: PdfWriter,
    node: Dict[str, Any],
    parent_ref: Any,
    kids_array: ArrayObject,
    page_refs: Dict[int, Any],
    parent_tree_entries: Dict[int, ArrayObject],
    page_xobject_refs: Dict[int, Dict[int, Any]],
    page_annotation_refs: Dict[int, Dict[str, Any]],
) -> Any:
    elem = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/StructElem"),
            NameObject("/S"): NameObject(f"/{_pdf_structure_tag(node.get('tag', 'P'))}"),
            NameObject("/P"): parent_ref,
        }
    )

    page_no = int(node.get("page") or 0)
    tag = str(node.get("tag", "")).strip()
    if page_no > 0 and page_no in page_refs and tag != "Document":
        elem[NameObject("/Pg")] = page_refs[page_no]

    text = (node.get("text") or "").strip()
    if text:
        elem[NameObject("/Alt")] = TextStringObject(text)
        elem[NameObject("/ActualText")] = TextStringObject(text)
    elif tag == "Figure":
        elem[NameObject("/Alt")] = TextStringObject("Figure")
    elif str(node.get("tag", "")).strip() == "Document":
        elem[NameObject("/Alt")] = TextStringObject("Document")

    elem_ref = writer._add_object(elem)
    kids_array.append(elem_ref)

    children = node.get("children", []) or []
    link_xref = node.get("link_xref")
    link_annot_ref = None
    if page_no in page_annotation_refs and link_xref is not None:
        link_annot_ref = page_annotation_refs[page_no].get(f"id:{int(link_xref)}")
    link_part_mcids = node.get("_link_part_mcids") or []

    if children or link_annot_ref is not None or link_part_mcids:
        child_refs = ArrayObject()
        if link_annot_ref is not None:
            child_refs.append(
                writer._add_object(
                    DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/OBJR"),
                            NameObject("/Pg"): page_refs[page_no],
                            NameObject("/Obj"): link_annot_ref,
                        }
                    )
                )
            )
        if link_part_mcids and page_no in page_refs:
            for part in link_part_mcids:
                mcr_ref = writer._add_object(
                    DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/MCR"),
                            NameObject("/MCID"): NumberObject(int(part["mcid"])),
                            NameObject("/Pg"): page_refs[page_no],
                        }
                    )
                )
                child_refs.append(mcr_ref)
                parent_tree_entries.setdefault(page_no, ArrayObject())
                parent_tree_entries[page_no].append(elem_ref)
        for child in children:
            _append_structure_node(
                writer,
                child,
                elem_ref,
                child_refs,
                page_refs,
                parent_tree_entries,
                page_xobject_refs,
                page_annotation_refs,
            )
        elem[NameObject("/K")] = child_refs
    elif "_mcid" in node and page_no in page_refs:
        mcr = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/MCR"),
                NameObject("/MCID"): NumberObject(int(node["_mcid"])),
                NameObject("/Pg"): page_refs[page_no],
            }
        )
        mcr_ref = writer._add_object(mcr)
        elem[NameObject("/K")] = mcr_ref
        parent_tree_entries.setdefault(page_no, ArrayObject())
        parent_tree_entries[page_no].append(elem_ref)
    else:
        elem[NameObject("/K")] = ArrayObject()

    return elem_ref
# === Lines 990-999 ===



def create_tagged_pdf(pdf_bytes: bytes, tree: Dict[str, Any], output_path: str) -> str:
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    writer = PdfWriter(clone_from=reader)

    root = writer.root_object
    root[NameObject("/MarkInfo")] = DictionaryObject({NameObject("/Marked"): BooleanObject(True)})

    role_map = DictionaryObject(
        {
            NameObject("/Reference"): NameObject("/BibEntry"),
        }
    )
    struct_root = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/StructTreeRoot"),
            NameObject("/K"): ArrayObject(),
            NameObject("/ParentTree"): DictionaryObject({NameObject("/Nums"): ArrayObject()}),
            NameObject("/ParentTreeNextKey"): NumberObject(0),
            NameObject("/RoleMap"): role_map,
        }
    )
    struct_root_ref = writer._add_object(struct_root)
    root[NameObject("/StructTreeRoot")] = struct_root_ref

    page_refs = {
        page_index + 1: writer.pages[page_index].indirect_reference
        for page_index in range(len(writer.pages))
    }
    page_xobject_refs = {
        page_index + 1: _build_page_xobject_map(writer.pages[page_index])
        for page_index in range(len(writer.pages))
    }

    next_mcid_by_page: Dict[int, int] = {}
    _assign_mcids(tree, next_mcid_by_page)

    parent_tree_entries: Dict[int, ArrayObject] = {}
    _append_structure_node(
        writer,
        tree,
        struct_root_ref,
        struct_root[NameObject("/K")],
        page_refs,
        parent_tree_entries,
        page_xobject_refs,
        {
            page_index + 1: _build_page_annotation_map(writer.pages[page_index])
            for page_index in range(len(writer.pages))
        },
    )

    marked_nodes = _iter_marked_content_nodes(tree)
    nodes_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for node in marked_nodes:
        page_no = int(node.get("page") or 0)
        if page_no > 0:
            nodes_by_page.setdefault(page_no, []).append(node)

    parent_tree_nums = ArrayObject()
    for page_index, page in enumerate(writer.pages):
        page_no = page_index + 1
        nodes = nodes_by_page.get(page_no, [])
        if nodes:
            font_name, gs_name = _ensure_overlay_resources(writer, page)
            overlay = _build_page_overlay_stream(page, nodes, font_name, gs_name)
            if overlay:
                _append_overlay_stream(writer, page, overlay)

        page[NameObject("/StructParents")] = NumberObject(page_index)
        entry_array = parent_tree_entries.get(page_no, ArrayObject())
        while len(entry_array) < next_mcid_by_page.get(page_no, 0):
            entry_array.append(NullObject())
        parent_tree_nums.append(NumberObject(page_index))
        parent_tree_nums.append(writer._add_object(entry_array))

    parent_tree = DictionaryObject({NameObject("/Nums"): parent_tree_nums})
    struct_root[NameObject("/ParentTree")] = writer._add_object(parent_tree)
    struct_root[NameObject("/ParentTreeNextKey")] = NumberObject(len(writer.pages))
    root_k = struct_root.get("/K")
    if isinstance(root_k, ArrayObject) and len(root_k) == 1:
        struct_root[NameObject("/K")] = root_k[0]

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file_obj:
        writer.write(file_obj)

    return str(target)
