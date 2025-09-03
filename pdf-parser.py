from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import argparse
import logging
import json
import re
import unicodedata
from time import perf_counter
import pdfplumber
import pandas as pd


# Module-level logger (align with main.py style)
logger = logging.getLogger(__name__)


class PerfTracker:
    def __init__(self) -> None:
        self._t0: float = perf_counter()
        self.metrics: Dict[str, Dict[str, float]] = {}

    class _Timer:
        def __init__(self, tracker: 'PerfTracker', name: str) -> None:
            self.tracker = tracker
            self.name = name
            self.dt: float = 0.0
            self._start: float = 0.0

        def __enter__(self) -> 'PerfTracker._Timer':
            self._start = perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.dt = perf_counter() - self._start
            self.tracker.add(self.name, self.dt)
            logger.info(f"perf | {self.name} took {self.dt:.3f}s")

    def timer(self, name: str) -> 'PerfTracker._Timer':
        return PerfTracker._Timer(self, name)

    def add(self, name: str, seconds: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = {"total": 0.0, "count": 0.0}
        self.metrics[name]["total"] += float(seconds)
        self.metrics[name]["count"] += 1.0

    def total_elapsed(self) -> float:
        return perf_counter() - self._t0

    def log_summary(self, header: str = "Performance summary") -> None:
        logger.info("=" * 60)
        logger.info(header)
        for name, data in sorted(self.metrics.items()):
            total = data.get("total", 0.0)
            count = int(data.get("count", 0.0))
            avg = (total / count) if count else total
            logger.info(f"perf | {name}: total={total:.3f}s count={count} avg={avg:.3f}s")
        logger.info(f"perf | total_analysis_time: {self.total_elapsed():.3f}s")
        logger.info("=" * 60)


@dataclass
class TableRecord:
    page: int
    bbox: List[float]
    engine: str
    confidence: Optional[float]
    header: List[str]
    rows: List[List[Optional[str]]]


def normalize_header(cells: List[Optional[str]]) -> List[str]:
    normalized: List[str] = []
    for cell in cells:
        if cell is None:
            normalized.append("")
            continue
        text = str(cell).strip()
        text = " ".join(text.split())
        normalized.append(text.lower())
    return normalized


def headers_match(h1: List[str], h2: List[str]) -> bool:
    if not h1 or not h2:
        return False
    # Simple heuristic: same length and majority of columns equal
    if len(h1) != len(h2):
        return False
    matches = sum(1 for a, b in zip(h1, h2) if a == b and a != "")
    return matches >= max(1, int(0.6 * len(h1)))


def extract_tables_from_page(page: pdfplumber.page.Page) -> List[TableRecord]:
    records: List[TableRecord] = []
    try:
        # Try pdfplumber's basic table extraction; tweak settings if needed later
        tables = page.extract_tables()
        for tbl in tables or []:
            if not tbl:
                continue
            # Smart header detection: if first row is mostly empty or not a good header,
            # search for a later row that maps to known antenna columns.
            def clean_row(r: List[Optional[str]]) -> List[Optional[str]]:
                return [None if c is None else (" ".join(str(c).split())) for c in r]

            cleaned_tbl = [clean_row(r) for r in tbl]
            header_idx = 0
            candidate_scores: List[Tuple[int, int]] = []
            max_scan = min(10, len(cleaned_tbl))
            for i in range(max_scan):
                norm = normalize_header([str(x) if x is not None else "" for x in cleaned_tbl[i]])
                mapped = [map_header_to_canonical(h) for h in norm]
                score = len([m for m in mapped if m])
                non_empty = len([h for h in norm if h])
                # Prefer rows with at least 3 mapped columns and some non-empties
                if score >= 3 and non_empty >= 3:
                    candidate_scores.append((i, score))
            if candidate_scores:
                # Choose highest score, earliest index on tie
                candidate_scores.sort(key=lambda x: (-x[1], x[0]))
                header_idx = candidate_scores[0][0]

            header = normalize_header([str(x) if x is not None else "" for x in cleaned_tbl[header_idx]]) if cleaned_tbl else []
            rows = [clean_row(r) for r in cleaned_tbl[header_idx + 1 :]]
            # Fallback: if header looks valid but no rows extracted, try word-grid reconstruction
            mapped = [map_header_to_canonical(h) for h in header]
            if not rows and sum(1 for m in mapped if m) >= 3:
                try:
                    fallback_rows = reconstruct_rows_from_words(page, header)
                    if fallback_rows:
                        rows = fallback_rows
                        logger.info(f"Reconstructed {len(rows)} row(s) from words for page {page.page_number}")
                except Exception as ee:
                    logger.warning(f"Fallback reconstruction failed on page {page.page_number}: {ee}")
            bbox = list(page.bbox) if hasattr(page, "bbox") else [0, 0, 0, 0]
            records.append(
                TableRecord(
                    page=page.page_number,
                    bbox=bbox,
                    engine="pdfplumber",
                    confidence=None,
                    header=header,
                    rows=rows,
                )
            )
    except Exception as e:
        logger.warning(f"Failed to extract tables on page {page.page_number}: {e}")
    return records


def reconstruct_rows_from_words(page: pdfplumber.page.Page, header_cells: List[str]) -> List[List[Optional[str]]]:
    # Extract words with positions
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return []
    # Normalize words and compute centers
    norm_words = []
    for w in words:
        text = normalize_text_basic(w.get("text", ""))
        if not text:
            continue
        x_center = (float(w["x0"]) + float(w["x1"])) / 2.0
        y_center = (float(w["top"]) + float(w["bottom"])) / 2.0
        norm_words.append({"text": text, "x0": float(w["x0"]), "x1": float(w["x1"]), "xc": x_center, "yc": y_center, "top": float(w["top"]), "bottom": float(w["bottom"])})
    if not norm_words:
        return []
    # Identify header band: pick y where most header tokens appear
    header_tokens = []
    for cell in header_cells:
        cell_norm = normalize_text_basic(cell)
        if not cell_norm:
            continue
        header_tokens.extend(cell_norm.split())
    header_tokens = [t for t in header_tokens if t]
    if not header_tokens:
        return []
    # Build histogram of words matching header tokens by y bins
    y_counts: Dict[int, int] = {}
    bin_height = 4  # pixels
    for w in norm_words:
        if w["text"] in header_tokens:
            y_bin = int(w["yc"] // bin_height)
            y_counts[y_bin] = y_counts.get(y_bin, 0) + 1
    if not y_counts:
        return []
    header_bin = max(y_counts.items(), key=lambda kv: kv[1])[0]
    header_yc = header_bin * bin_height + bin_height / 2
    # Approximate header band range
    band_top = header_yc - 8
    band_bottom = header_yc + 8
    header_line_words = [w for w in norm_words if band_top <= w["yc"] <= band_bottom]
    if not header_line_words:
        # Fallback: scan lines to find a header-like line by keywords
        # Group words into lines by y proximity
        lines: List[List[Dict[str, Any]]] = []
        norm_words_sorted = sorted(norm_words, key=lambda w: w["yc"])  # type: ignore[name-defined]
        line_tol = 6
        current_line: List[Dict[str, Any]] = []
        current_y = None
        for w in norm_words_sorted:
            if current_y is None or abs(w["yc"] - current_y) <= line_tol:
                (current_line if current_y is not None else current_line).append(w)
                current_y = w["yc"] if current_y is None else current_y
            else:
                if current_line:
                    lines.append(sorted(current_line, key=lambda x: x["xc"]))
                current_line = [w]
                current_y = w["yc"]
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x["xc"]))

        def score_header_line(line_words: List[Dict[str, Any]]) -> int:
            tokens = {lw["text"] for lw in line_words}
            keywords_sets = [
                {"celdaid", "celda", "id"},
                {"rango", "consulta"},
                {"direccion", "dir"},
                {"loc", "localidad"},
                {"prov", "provincia"},
                {"rad", "km", "cob"},
                {"azimuth", "azimut"},
                {"lat"},
                {"long", "lon", "lng"},
                {"horiz", "horizontal"},
                {"vert", "vertical"},
            ]
            score = 0
            for ks in keywords_sets:
                if any(k in tokens for k in ks):
                    score += 1
            return score

        scored = [(i, score_header_line(ln)) for i, ln in enumerate(lines)]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= 3:
            header_line_words = lines[scored[0][0]]
            header_yc = sum(w["yc"] for w in header_line_words) / len(header_line_words)
            band_top = header_yc - 8
            band_bottom = header_yc + 8
        else:
            return []
    # Determine page size and set footer margin to filter footer rows
    try:
        page_height = float(getattr(page, "height", page.bbox[3]))
    except Exception:
        page_height = 1000.0
    footer_margin = 40.0

    # Helper to pick a distinctive token from the header cell (prefer last non-generic)
    GENERIC_TOKENS = {"celda", "cell", "id", "de", "consulta", "rango", "(km)", "km", "a.", "a", "(h)", "(v)"}
    def choose_key_token(cell_norm: str) -> str:
        toks = [t for t in cell_norm.split() if t]
        # Prefer tokens not in generic set; try from right to left
        for t in reversed(toks):
            if t not in GENERIC_TOKENS:
                return t
        # Fallback to last token
        return toks[-1] if toks else ""

    # For each header cell phrase, find span (x0, x1) on the header line using token sequence
    def find_phrase_span(cell_norm: str) -> Optional[Tuple[float, float, float]]:
        toks = [t for t in cell_norm.split() if t]
        if not toks:
            return None
        span_words = []
        pos = 0
        for tok in toks:
            found = None
            for i in range(pos, len(header_line_words)):
                hw = header_line_words[i]
                if hw["text"] == tok or tok in hw["text"] or hw["text"] in tok:
                    found = (i, hw)
                    break
            if found is None:
                # token not found; fall back to key token only
                key = choose_key_token(cell_norm)
                cands = [w for w in header_line_words if w["text"] == key or key in w["text"]]
                if not cands:
                    return None
                w = sorted(cands, key=lambda w: w["xc"])[0]
                return (float(w["x0"]), float(w["x1"]), float(w["xc"]))
            span_words.append(found[1])
            pos = found[0] + 1
        x0 = min(float(w["x0"]) for w in span_words)
        x1 = max(float(w["x1"]) for w in span_words)
        xc = (x0 + x1) / 2.0
        return (x0, x1, xc)

    column_extents: List[Tuple[int, float, float, float, str]] = []  # (idx, x0, x1, xc, norm_text)
    for idx, cell in enumerate(header_cells):
        cell_norm = normalize_text_basic(cell)
        if not cell_norm:
            continue
        span = find_phrase_span(cell_norm)
        if span is None:
            continue
        x0, x1, xc = span
        # Expand spans per column semantics
        expand_left = 6.0
        expand_right = 6.0
        # Widen address column more to the right so trailing house numbers stay inside
        if "direccion" in cell_norm:
            expand_right += 10.0
        # Keep city (loc) tight to avoid absorbing province
        if ("loc" in cell_norm) or ("localidad" in cell_norm):
            expand_left = max(4.0, expand_left - 2.0)
            expand_right = max(4.0, expand_right - 2.0)
        # Province and coverage columns stay tight
        if cell_norm.startswith("prov") or ("provincia" in cell_norm) or ("rad" in cell_norm and ("km" in cell_norm or "cob" in cell_norm)):
            expand_left = max(3.0, expand_left - 3.0)
            expand_right = max(3.0, expand_right - 3.0)
        x0 -= expand_left
        x1 += expand_right
        column_extents.append((idx, x0, x1, xc, cell_norm))
    if len(column_extents) < 2:
        logger.info("reconstruct: insufficient column extents; will fallback to token-center method")
        return _reconstruct_rows_simple(page, header_cells, header_line_words, band_bottom)
    # Sort columns left to right
    column_extents.sort(key=lambda t: t[3])
    # Build boundaries between adjacent columns; prefer gap between spans if non-overlapping
    boundaries: List[float] = []
    for i in range(len(column_extents) - 1):
        left = column_extents[i]
        right = column_extents[i + 1]
        # Use gap between span edges when non-overlapping; otherwise centers
        if right[1] > left[2]:
            mid = (left[2] + right[1]) / 2.0
        else:
            mid = (left[3] + right[3]) / 2.0
        boundaries.append(mid)
    # No longer using column_positions; extents already validated above
    # Extend boundaries to table edges based on header spans
    x_min = min(x0 for (_, x0, _, _, _) in column_extents)
    x_max = max(x1 for (_, _, x1, _, _) in column_extents)
    boundaries = [x_min - 1.0] + boundaries + [x_max + 1.0]
    # Collect body words below header band and above footer margin
    body_words = [w for w in norm_words if (w["yc"] > band_bottom + 2) and (w["yc"] < (page_height - footer_margin))]
    if not body_words:
        logger.info("reconstruct: no body words found under header band; using simple fallback")
        return _reconstruct_rows_simple(page, header_cells, header_line_words, band_bottom)
    # Cluster by y (rows)
    body_words.sort(key=lambda w: w["yc"])
    rows: List[List[Optional[str]]] = []
    current_y = None
    current_cells: List[List[str]] = [[] for _ in header_cells]
    y_tol = 4
    def flush_current():
        if current_y is None:
            return
        # Join cell tokens
        row_vals: List[Optional[str]] = []
        for tokens in current_cells:
            if tokens:
                row_vals.append(" ".join(tokens))
            else:
                row_vals.append(None)
        rows.append(row_vals)

    # Use boundaries first, with content-aware fixes for range/province
    col_centers = [xc for (_, _, _, xc, _) in column_extents]
    # Identify important column indexes by header text
    def find_col_index(predicate):
        for idx, (_i, _x0, _x1, _xc, txt) in enumerate(column_extents):
            if predicate(txt):
                return idx
        return None

    # No range column handling
    idx_address = find_col_index(lambda t: ("direccion" in t))
    idx_city = find_col_index(lambda t: ("loc" in t or "localidad" in t))
    idx_province = find_col_index(lambda t: (t.startswith("prov") or "provincia" in t))
    idx_coverage = find_col_index(lambda t: ("rad" in t and ("km" in t or "cob" in t)))

    def is_number_like(s: str) -> bool:
        return bool(re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?", s))

    for w in body_words:
        if current_y is None:
            current_y = w["yc"]
            current_cells = [[] for _ in header_cells]
        elif abs(w["yc"] - current_y) > y_tol:
            flush_current()
            current_y = w["yc"]
            current_cells = [[] for _ in header_cells]
        # Determine column by strict acceptance window (span) first
        col_idx = None
        for idx_e, (_i, ax0, ax1, _xc, _txt) in enumerate(column_extents):
            if ax0 <= w["xc"] <= ax1:
                col_idx = idx_e
                break
        # Then by boundaries if not inside any span
        for bi in range(len(boundaries) - 1):
            if col_idx is None and boundaries[bi] <= w["xc"] < boundaries[bi + 1]:
                col_idx = bi
                break
        # Fallback to nearest center if outside boundaries (rare)
        if col_idx is None and col_centers:
            diffs = [abs(w["xc"] - c) for c in col_centers]
            col_idx = diffs.index(min(diffs))

        # Content-aware fixes: ignore range; fix province vs coverage
        token = w["text"]
        # No special handling for range anymore (ignored)
        if idx_province is not None and idx_coverage is not None and col_idx == idx_province and is_number_like(token):
            col_idx = idx_coverage
        if col_idx is None:
            continue
        if col_idx >= len(current_cells):
            # Expand to match header len if needed
            missing = col_idx - len(current_cells) + 1
            current_cells.extend([[] for _ in range(missing)])
        current_cells[col_idx].append(w["text"])
    # Flush last row
    flush_current()
    # Trim trailing completely empty rows
    trimmed = []
    for r in rows:
        if any(cell for cell in r if cell and str(cell).strip() != ""):
            trimmed.append(r)
    if not trimmed:
        logger.info("reconstruct: phrase-span assignment produced 0 rows; trying token-center fallback")
        return _reconstruct_rows_simple(page, header_cells, header_line_words, band_bottom)
    return trimmed


def _reconstruct_rows_simple(page: pdfplumber.page.Page, header_cells: List[str], header_line_words: List[Dict[str, Any]], band_bottom: float) -> List[List[Optional[str]]]:
    # Simpler approach: use first distinctive token center per header cell and midpoint boundaries
    words_all = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words_all:
        return []
    norm_words = []
    for w in words_all:
        text = normalize_text_basic(w.get("text", ""))
        if not text:
            continue
        x_center = (float(w["x0"]) + float(w["x1"])) / 2.0
        y_center = (float(w["top"]) + float(w["bottom"])) / 2.0
        norm_words.append({"text": text, "x0": float(w["x0"]), "x1": float(w["x1"]), "xc": x_center, "yc": y_center, "top": float(w["top"]), "bottom": float(w["bottom"])})
    if not norm_words:
        return []
    # Column centers
    GENERIC_TOKENS = {"celda", "cell", "id", "de", "consulta", "rango", "(km)", "km", "a.", "a", "(h)", "(v)"}
    def choose_key_token(cell_norm: str) -> str:
        toks = [t for t in cell_norm.split() if t]
        for t in reversed(toks):
            if t not in GENERIC_TOKENS:
                return t
        return toks[-1] if toks else ""
    col_centers: List[float] = []
    for cell in header_cells:
        cn = normalize_text_basic(cell)
        if not cn:
            continue
        key = choose_key_token(cn)
        cands = [w for w in header_line_words if w["text"] == key] or [w for w in header_line_words if key in w["text"]]
        if not cands:
            continue
        col_centers.append(sorted(cands, key=lambda w: w["xc"])[0]["xc"])
    if len(col_centers) < 2:
        return []
    col_centers.sort()
    boundaries = []
    for i in range(len(col_centers) - 1):
        boundaries.append((col_centers[i] + col_centers[i + 1]) / 2.0)
    # to edges
    x_min = min(w["x0"] for w in norm_words)
    x_max = max(w["x1"] for w in norm_words)
    boundaries = [x_min - 1.0] + boundaries + [x_max + 1.0]
    # Body words only
    body_words = [w for w in norm_words if w["yc"] > band_bottom + 2]
    body_words.sort(key=lambda w: w["yc"])
    rows: List[List[Optional[str]]] = []
    current_y = None
    current_cells: List[List[str]] = [[] for _ in header_cells]
    y_tol = 5
    def flush_current():
        if current_y is None:
            return
        row_vals: List[Optional[str]] = []
        for tokens in current_cells:
            row_vals.append(" ".join(tokens) if tokens else None)
        # Keep non-empty only
        if any(v and v.strip() for v in row_vals):
            rows.append(row_vals)
    for w in body_words:
        if current_y is None or abs(w["yc"] - current_y) > y_tol:
            flush_current()
            current_y = w["yc"]
            current_cells = [[] for _ in header_cells]
        # boundaries assignment
        col_idx = None
        for bi in range(len(boundaries) - 1):
            if boundaries[bi] <= w["xc"] < boundaries[bi + 1]:
                col_idx = bi
                break
        if col_idx is None:
            continue
        if col_idx >= len(current_cells):
            missing = col_idx - len(current_cells) + 1
            current_cells.extend([[] for _ in range(missing)])
        current_cells[col_idx].append(w["text"])
    flush_current()
    return rows


def merge_tables_across_pages(tables: List[TableRecord]) -> List[TableRecord]:
    if not tables:
        return []
    merged: List[TableRecord] = []
    for tbl in tables:
        placed = False
        for m in merged:
            if headers_match(m.header, tbl.header):
                # Append rows; keep earliest page/bbox as representative
                m.rows.extend(tbl.rows)
                placed = True
                break
        if not placed:
            merged.append(tbl)
    return merged


def consolidate_tables_by_header(tables: List[TableRecord]) -> List[TableRecord]:
    """Globally merge tables that share the exact normalized header.
    Keeps the earliest page and bbox, appends rows from all tables in the group.
    """
    if not tables:
        return []
    groups: Dict[tuple, TableRecord] = {}
    for tbl in tables:
        key = tuple(tbl.header)
        if key not in groups:
            groups[key] = TableRecord(
                page=tbl.page,
                bbox=tbl.bbox,
                engine=tbl.engine,
                confidence=tbl.confidence,
                header=list(tbl.header),
                rows=list(tbl.rows),
            )
        else:
            # Keep earliest page as representative
            if tbl.page < groups[key].page:
                groups[key].page = tbl.page
                groups[key].bbox = tbl.bbox
            groups[key].rows.extend(tbl.rows)
    return list(groups.values())


def write_outputs(pdf_path: Path, output_dir: Path, tables: List[TableRecord]) -> None:
    output_base = output_dir / pdf_path.stem
    output_base.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_base / "tables.json"
    json_payload: Dict[str, Any] = {
        "file": str(pdf_path),
        "num_tables": len(tables),
        "tables": [asdict(t) for t in tables],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote JSON: {json_path}")

    # CSV output (concatenate with header separators)
    # Emit one CSV containing all merged tables, separated by blank lines
    csv_path = output_base / "tables.csv"
    frames: List[pandas.core.frame.DataFrame] = []  # type: ignore[name-defined]
    for t in tables:
        if not t.header:
            continue
        # Build unique, string-only column names to avoid pandas InvalidIndexError
        base_cols = [str(h) if h else f"col_{i+1}" for i, h in enumerate(t.header)]
        seen: Dict[str, int] = {}
        unique_cols: List[str] = []
        for name in base_cols:
            count = seen.get(name, 0)
            seen[name] = count + 1
            if count == 0:
                unique_cols.append(name)
            else:
                unique_cols.append(f"{name}_{count+1}")
        df = pd.DataFrame(t.rows, columns=unique_cols)
        # Add metadata columns for traceability
        df.insert(0, "source_page", t.page)
        frames.append(df)
    if frames:
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined.to_csv(csv_path, index=False)
        logger.info(f"Wrote CSV: {csv_path}")
    else:
        # Write empty CSV to indicate no tables found
        pd.DataFrame().to_csv(csv_path, index=False)
        logger.info(f"No tables found. Wrote empty CSV: {csv_path}")


def normalize_text_basic(value: str) -> str:
    # Lowercase, strip, remove multiple spaces, remove accents
    text = value.lower().strip()
    text = " ".join(text.split())
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text


def parse_numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "-", "--"}:
        return None
    # Remove units and symbols
    s = s.replace("°", "").replace("º", "")
    s = re.sub(r"(?i)\bdeg\b", "", s)
    s = re.sub(r"(?i)dbm", "", s)
    s = s.replace("dB", "")
    # Normalize decimal separators: handle European format (e.g., 1.234,56)
    # Remove spaces
    s = s.replace(" ", "")
    # If there are both dots and commas, assume dot is thousands sep and comma is decimal
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # If only comma, treat as decimal sep
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        # If only dots, leave as is (could be thousands, but ambiguous)
    # Remove any non-numeric except leading minus and dot
    s = re.sub(r"[^0-9.-]", "", s)
    try:
        return float(s)
    except Exception:
        return None


CANONICAL_ANTENNA_COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
    "azimuth_deg",
    "horizontal_aperture_deg",
    "vertical_aperture_deg",
    "coverage_radius_km",
    "signal_strength_dbm",
    "address",
    "city",
    "province",
]


ANTENNA_HEADER_SYNONYMS: Dict[str, List[str]] = {
    "cell_id": [
        "cell id", "cellid", "cell-id", "id celda", "id de celda", "celda id", "cid", "celdaid",
    ],
    "latitude": [
        "latitud", "latitude", "lat.", "lat",
    ],
    "longitude": [
        "longitud", "longitude", "long.", "lon", "lng",
    ],
    "azimuth_deg": [
        "azimut", "azimuth", "acimut", "aci\u0301mut", "azi", "acimutal",
    ],
    "horizontal_aperture_deg": [
        "apertura horizontal", "apertura h", "apertura (h)", "ancho haz horizontal", "beamwidth horizontal", "apertura hor", "a. horiz", "a. h", "a h",
    ],
    "vertical_aperture_deg": [
        "apertura vertical", "apertura v", "apertura (v)", "ancho haz vertical", "beamwidth vertical", "apertura vert", "a. vert", "a. v", "a v",
    ],
    "coverage_radius_km": [
        "rad cob", "rad cob (km)", "radio de cobertura", "radio cobertura", "radio (km)", "coverage radius", "radius (km)",
    ],
    "signal_strength_dbm": [
        "potencia", "nivel senal", "nivel se\u00f1al", "intensidad senal", "intensidad se\u00f1al", "rsrp", "rssi", "signal strength", "dbm",
    ],
    "address": [
        "direccion", "celda direccion", "dir", "domicilio", "calle", "celda calle altura", "calle altura",
    ],
    "city": [
        "ciudad", "localidad", "loc", "celda loc",
    ],
    "province": [
        "provincia", "prov", "celda prov",
    ],
}


def map_header_to_canonical(header_cell: Optional[str]) -> Optional[str]:
    if header_cell is None:
        return None
    base = normalize_text_basic(str(header_cell))
    base = base.replace(":", "").replace("/", " ")
    base = re.sub(r"\s+", " ", base)
    # Direct exact matches
    for canonical, variants in ANTENNA_HEADER_SYNONYMS.items():
        if base in [normalize_text_basic(v) for v in variants]:
            return canonical
    # Fallback: substring checks for common tokens
    if "lat" in base and "latitud" in normalize_text_basic("latitud"):
        return "latitude"
    if any(tok in base for tok in ["long", "lng"]):
        return "longitude"
    if "azim" in base or "acim" in base:
        return "azimuth_deg"
    if ("apertura" in base or "beamwidth" in base) and ("h" in base or "hori" in base):
        return "horizontal_aperture_deg"
    if ("apertura" in base or "beamwidth" in base) and ("v" in base or "vert" in base):
        return "vertical_aperture_deg"
    if ("rad" in base and "cob" in base) or ("radio" in base and ("km" in base or "cob" in base)):
        return "coverage_radius_km"
    if "direccion" in base or base.startswith("dir"):
        return "address"
    if "localidad" in base or base.startswith("loc") or "ciudad" in base:
        return "city"
    if base.startswith("prov") or "provincia" in base:
        return "province"
    if "estado" in base or base == "state":
        return "state"
    if any(tok in base for tok in ["rsrp", "rssi", "dbm", "senal", "señal", "potencia"]):
        return "signal_strength_dbm"
    if "celda" in base or "cell" in base or base == "cid":
        return "cell_id"
    return None


def table_to_dataframe_if_antenna(table: TableRecord) -> Optional[pd.DataFrame]:
    if not table.header:
        return None
    mapped = [map_header_to_canonical(h) for h in table.header]
    # Keep columns that map to known canonical names (deduplicate while preserving order)
    keep_indices: List[int] = []
    keep_names: List[str] = []
    seen: set = set()
    for idx, name in enumerate(mapped):
        if name and name not in seen:
            keep_indices.append(idx)
            keep_names.append(name)
            seen.add(name)
    # Require at least 3 of the target columns to qualify
    if len(keep_names) < 3:
        return None
    # Build dataframe
    reduced_rows: List[List[Optional[str]]] = []
    for row in table.rows:
        reduced_rows.append([row[i] if i < len(row) else None for i in keep_indices])
    df = pd.DataFrame(reduced_rows, columns=keep_names)
    # Normalize numeric columns where applicable
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].map(parse_numeric)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].map(parse_numeric)
    if "azimuth_deg" in df.columns:
        df["azimuth_deg"] = df["azimuth_deg"].map(parse_numeric)
    if "horizontal_aperture_deg" in df.columns:
        df["horizontal_aperture_deg"] = df["horizontal_aperture_deg"].map(parse_numeric)
    if "vertical_aperture_deg" in df.columns:
        df["vertical_aperture_deg"] = df["vertical_aperture_deg"].map(parse_numeric)
    if "coverage_radius_km" in df.columns:
        df["coverage_radius_km"] = df["coverage_radius_km"].map(parse_numeric)
    if "signal_strength_dbm" in df.columns:
        df["signal_strength_dbm"] = df["signal_strength_dbm"].map(parse_numeric)
    # Add metadata
    df.insert(0, "source_page", table.page)
    return df


def extract_antennas_from_tables(tables: List[TableRecord]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for t in tables:
        df = table_to_dataframe_if_antenna(t)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["source_page"] + CANONICAL_ANTENNA_COLUMNS)
    combined = pd.concat(frames, axis=0, ignore_index=True)
    # Ensure all canonical columns exist
    for col in CANONICAL_ANTENNA_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    # Order columns
    ordered_cols = ["source_page"] + CANONICAL_ANTENNA_COLUMNS
    combined = combined[ordered_cols]
    # Drop rows entirely empty across canonical columns
    non_empty_mask = combined[CANONICAL_ANTENNA_COLUMNS].notna().any(axis=1)
    combined = combined[non_empty_mask]
    return combined


def extract_antennas_claro(tables: List[TableRecord]) -> pd.DataFrame:
    # Claro headers: build a single base table from all address/geo rows, and a document-level
    # orientation series from azimuth/aperture-only chunks. Then map orientations to base rows.
    base_frames: List[pd.DataFrame] = []
    orientation_pairs: List[Tuple[Optional[float], Optional[float]]] = []
    roster_frames: List[pd.DataFrame] = []

    for t in tables:
        header = [h or "" for h in t.header]
        hn = [normalize_text_basic(h) for h in header]
        text_all = " ".join(hn)
        has_base = any(tok in text_all for tok in ["calle", "numero", "lat", "long", "radio", "cobertura", "km"])
        has_orient = ("azimut" in text_all or "angulo" in text_all or "apertura" in text_all) and not has_base
        has_roster = ("pais" in text_all and "localidad" in text_all and ("departamento" in text_all or "depto" in text_all or "partido" in text_all) and "provincia" in text_all)

        df_raw = pd.DataFrame(t.rows, columns=[f"c{i}" for i in range(len(header))])

        if has_base:
            def find_col(keys: List[str]) -> Optional[int]:
                for idx_h, h in enumerate(hn):
                    if any(k in h for k in keys):
                        return idx_h
                return None
            idx_calle = find_col(["calle"]) 
            idx_numero = find_col(["numero"]) 
            idx_lat = find_col(["lat"]) 
            idx_long = find_col(["long"]) 
            idx_radio = find_col(["radio", "cobertura", "km"]) 

            proj = pd.DataFrame({
                "source_page": t.page,
                "address": None,
                "latitude": None,
                "longitude": None,
                "coverage_radius_km": None,
            }, index=df_raw.index)

            if idx_calle is not None:
                calle = df_raw.iloc[:, idx_calle].astype(str).fillna("").str.strip()
                if idx_numero is not None:
                    numero = df_raw.iloc[:, idx_numero].astype(str).fillna("").str.strip()
                    addr = (calle + " " + numero).str.replace(r"\bnan\b", "", regex=True).str.strip()
                else:
                    addr = calle
                proj["address"] = addr.where(addr != "", None)
            if idx_lat is not None:
                proj["latitude"] = df_raw.iloc[:, idx_lat].map(parse_numeric)
            if idx_long is not None:
                proj["longitude"] = df_raw.iloc[:, idx_long].map(parse_numeric)
            if idx_radio is not None:
                proj["coverage_radius_km"] = df_raw.iloc[:, idx_radio].map(parse_numeric)

            base_frames.append(proj)

        elif has_orient:
            # Flatten numeric values row-wise into (azimuth, aperture) pairs in reading order
            for _, row in df_raw.iterrows():
                nums: List[float] = []
                for val in row.tolist():
                    num = parse_numeric(val if (val is not None and str(val).strip() != "") else None)
                    if num is not None:
                        nums.append(num)
                for i in range(0, len(nums) - 1, 2):
                    orientation_pairs.append((nums[i], nums[i + 1]))
        elif has_roster:
            # Capture roster rows (PAIS, LOCALIDAD, DEPARTAMENTO, PROVINCIA)
            # Try to find approximate column indices
            def find_col_idx(names: List[str]) -> Optional[int]:
                for idx_h, h in enumerate(hn):
                    if any(n in h for n in names):
                        return idx_h
                return None
            idx_pais = find_col_idx(["pais"]) or 0
            idx_loc = find_col_idx(["localidad", "localidad/ciudad"]) or 1
            idx_depto = find_col_idx(["departamento", "depto", "partido"]) or 2
            idx_prov = find_col_idx(["provincia"]) or 3
            rf = pd.DataFrame({
                "source_page": t.page,
                "pais": df_raw.iloc[:, idx_pais] if idx_pais < len(df_raw.columns) else None,
                "city": df_raw.iloc[:, idx_loc] if idx_loc < len(df_raw.columns) else None,
                "department": df_raw.iloc[:, idx_depto] if idx_depto < len(df_raw.columns) else None,
                "province": df_raw.iloc[:, idx_prov] if idx_prov < len(df_raw.columns) else None,
            })
            # Normalize text to plain strings, drop empty rows
            for col in ["pais", "city", "department", "province"]:
                if col in rf.columns:
                    rf[col] = rf[col].astype(str).map(lambda s: None if s is None or str(s).strip() == "" else " ".join(str(s).split()))
            keep_mask = rf[["pais", "city", "department", "province"]].notna().any(axis=1)
            rf = rf[keep_mask]
            if not rf.empty:
                roster_frames.append(rf)

    if not base_frames:
        return pd.DataFrame(columns=["source_page"] + CANONICAL_ANTENNA_COLUMNS)

    base = pd.concat(base_frames, axis=0, ignore_index=True)
    # Compute proportional mapping indices helper
    def proportional_map(total_target: int, total_source: int) -> List[int]:
        if total_source <= 0:
            return [0] * total_target
        if total_target <= 1:
            return [0]
        if total_source == 1:
            return [0] * total_target
        idxs = []
        for i in range(total_target):
            j = int(round(i * (total_source - 1) / (total_target - 1)))
            j = max(0, min(total_source - 1, j))
            idxs.append(j)
        return idxs

    # Prepare orientation vectors
    az_vec: List[Optional[float]] = []
    ap_vec: List[Optional[float]] = []
    if orientation_pairs:
        az_src = [p[0] for p in orientation_pairs]
        ap_src = [p[1] for p in orientation_pairs]
    else:
        az_src, ap_src = [], []

    # If there's a roster, anchor output size and fill city/province; else keep base size
    if roster_frames:
        roster = pd.concat(roster_frames, axis=0, ignore_index=True)
        R = len(roster)
        B = len(base)
        M = len(orientation_pairs)
        base_idx = proportional_map(R, B)
        orient_idx = proportional_map(R, M) if M > 0 else [0] * R
        # Build output aligned to roster
        rows = []
        for i in range(R):
            b = base.iloc[base_idx[i]] if B > 0 else pd.Series()
            az = az_src[orient_idx[i]] if M > 0 else None
            ap = ap_src[orient_idx[i]] if M > 0 else None
            rows.append({
                "source_page": b.get("source_page", None) if not b.empty else roster.at[i, "source_page"],
                "cell_id": None,
                "latitude": parse_numeric(b.get("latitude", None)) if not b.empty else None,
                "longitude": parse_numeric(b.get("longitude", None)) if not b.empty else None,
                "azimuth_deg": parse_numeric(az),
                "horizontal_aperture_deg": parse_numeric(ap),
                "vertical_aperture_deg": None,
                "coverage_radius_km": parse_numeric(b.get("coverage_radius_km", None)) if not b.empty else None,
                "signal_strength_dbm": None,
                "address": b.get("address", None) if not b.empty else None,
                "city": roster.at[i, "city"] if "city" in roster.columns else None,
                "province": roster.at[i, "province"] if "province" in roster.columns else None,
            })
        out = pd.DataFrame(rows)
    else:
        # No roster: keep base length and proportional map orientations
        B = len(base)
        M = len(orientation_pairs)
        orient_idx = proportional_map(B, M) if M > 0 else [0] * B
        az_col = [az_src[j] if M > 0 else None for j in orient_idx]
        ap_col = [ap_src[j] if M > 0 else None for j in orient_idx]
        out = pd.DataFrame({
            "source_page": base["source_page"],
            "cell_id": None,
            "latitude": base["latitude"].map(parse_numeric),
            "longitude": base["longitude"].map(parse_numeric),
            "azimuth_deg": pd.Series(az_col).map(parse_numeric),
            "horizontal_aperture_deg": pd.Series(ap_col).map(parse_numeric),
            "vertical_aperture_deg": None,
            "coverage_radius_km": base["coverage_radius_km"].map(parse_numeric),
            "signal_strength_dbm": None,
            "address": base["address"],
            "city": None,
            "province": None,
        })

    # No deduplication: return all rows to inspect full results

    for col in CANONICAL_ANTENNA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    ordered_cols = ["source_page"] + CANONICAL_ANTENNA_COLUMNS
    out = out[ordered_cols]
    return out


def _normalize_address_for_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = normalize_text_basic(str(value))
    return re.sub(r"\s+", " ", text)


def _round_coord(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    try:
        return round(float(value), ndigits) if value is not None else None
    except Exception:
        return None


def derive_antenna_sites(cells_df: pd.DataFrame) -> pd.DataFrame:
    if cells_df is None or cells_df.empty:
        return pd.DataFrame(columns=["source_page", "address", "city", "province", "latitude", "longitude"])
    sites = cells_df.copy()
    # Keys for site uniqueness
    sites["address_norm"] = sites.get("address").map(_normalize_address_for_key) if "address" in sites.columns else None
    sites["lat_r"] = sites.get("latitude").map(lambda v: _round_coord(v, 6)) if "latitude" in sites.columns else None
    sites["lon_r"] = sites.get("longitude").map(lambda v: _round_coord(v, 6)) if "longitude" in sites.columns else None

    def first_non_null(series: pd.Series):
        for v in series:
            if pd.notna(v):
                return v
        return None

    grouped = sites.groupby(["lat_r", "lon_r", "address_norm"], dropna=False, as_index=False).agg({
        "source_page": "min",
        "address": first_non_null,
        "city": first_non_null,
        "province": first_non_null,
        "latitude": first_non_null,
        "longitude": first_non_null,
    })
    return grouped[["source_page", "address", "city", "province", "latitude", "longitude"]]


def write_cells_and_antennas_outputs(pdf_path: Path, output_dir: Path, cells_df: pd.DataFrame) -> Tuple[Tuple[Path, Path], Tuple[Path, Path]]:
    output_base = output_dir / pdf_path.stem
    output_base.mkdir(parents=True, exist_ok=True)

    # Cells outputs
    cells_csv = output_base / "cells.csv"
    cells_json = output_base / "cells.json"
    cells_df.to_csv(cells_csv, index=False)
    cells_payload = {
        "file": str(pdf_path),
        "count": int(len(cells_df)),
        "records": cells_df.to_dict(orient="records"),
    }
    with cells_json.open("w", encoding="utf-8") as f:
        json.dump(cells_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote cells CSV: {cells_csv}")
    logger.info(f"Wrote cells JSON: {cells_json}")

    # Antenna sites (address/city/province/lat/lng)
    sites_df = derive_antenna_sites(cells_df)
    antennas_csv = output_base / "antennas.csv"
    antennas_json = output_base / "antennas.json"
    sites_df.to_csv(antennas_csv, index=False)
    ant_payload = {
        "file": str(pdf_path),
        "count": int(len(sites_df)),
        "records": sites_df.to_dict(orient="records"),
    }
    with antennas_json.open("w", encoding="utf-8") as f:
        json.dump(ant_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote antennas CSV: {antennas_csv}")
    logger.info(f"Wrote antennas JSON: {antennas_json}")

    return (cells_csv, cells_json), (antennas_csv, antennas_json)


def process_pdf(input_pdf: Path, output_dir: Path, extract: Optional[str] = None, provider: str = "auto") -> None:
    perf = PerfTracker()
    tables_all: List[TableRecord] = []

    # Open PDF timing
    pdf_file = None
    with perf.timer("open_pdf"):
        pdf_file = pdfplumber.open(str(input_pdf))

    with pdf_file as pdf:
        logger.info(f"Opened PDF with {len(pdf.pages)} pages: {input_pdf}")
        for page in pdf.pages:
            with perf.timer(f"page_{page.page_number}_extract") as t:
                page_tables = extract_tables_from_page(page)
            perf.add("extract_pages_total", t.dt)
            if page_tables:
                logger.info(f"Found {len(page_tables)} table(s) on page {page.page_number}")
            tables_all.extend(page_tables)

    # Merge across pages by header similarity (simple starter)
    with perf.timer("merge_tables"):
        merged_tables = merge_tables_across_pages(tables_all)
    logger.info(f"Merged into {len(merged_tables)} table group(s)")

    # Consolidate tables that share the same header across the document to keep counts consistent
    with perf.timer("consolidate_tables_by_header"):
        consolidated_tables = consolidate_tables_by_header(merged_tables)
    with perf.timer("write_tables_outputs"):
        write_outputs(input_pdf, output_dir, consolidated_tables)

    if extract == "antennas":
        with perf.timer("extract_antennas"):
            if provider == "claro":
                cells_df = extract_antennas_claro(consolidated_tables)
            else:
                cells_df = extract_antennas_from_tables(consolidated_tables)
        with perf.timer("write_antennas_outputs"):
            write_cells_and_antennas_outputs(input_pdf, output_dir, cells_df)

    # Final performance summary
    perf.log_summary()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tables from PDF into JSON and CSV")
    parser.add_argument("--input", required=True, help="Path to input PDF file")
    parser.add_argument("--output-dir", default="output", help="Directory for outputs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--extract", choices=["all", "antennas"], default="all", help="Optional specialized extraction")
    parser.add_argument("--provider", choices=["auto", "personal", "claro"], default="auto", help="Vendor/provider-specific parsing hints")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s'
    )

    input_pdf = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_pdf.exists():
        logger.error(f"Input file does not exist: {input_pdf}")
        raise SystemExit(1)

    extract_mode = args.extract if hasattr(args, "extract") else "all"
    process_pdf(input_pdf, output_dir, extract=extract_mode, provider=args.provider)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s - %(message)s')
        logger.info("Use --help to see command-line options")