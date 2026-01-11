#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract a specific table from a multi-page PDF and save as a semicolon-separated CSV.

Expected table header (exact columns, in Swedish):
  - Betalningsdag
  - Belopp
  - Betalningstyp
  - Mottagare/Betalare

The script:
- Finds the header words on each page
- Derives column x-boundaries from the header word boxes
- Assigns each word below the header into the right column by x-center
- Builds rows by clustering words with similar y (line) values
- Writes semicolon-separated CSV in UTF-8 with BOM

Debugging:
- Emits verbose 'DBG:' print lines about header detection, column bounds, line clustering, and row counts.

Usage:
  python parse_pdf_table.py input.pdf
"""

import sys
import re
from pathlib import Path
import pandas as pd

EXPECTED_HEADERS = ["Betalningsdag", "Belopp", "Betalningstyp", "Mottagare/Betalare"]
HEADER_LOWER = [h.lower() for h in EXPECTED_HEADERS]

SEMICOLON = ";"

# Tuning knobs for line grouping and header detection
LINE_Y_TOL = 2.0        # how tightly to group words into the same text line
ROW_GAP_ALLOW = 6.0     # gap in y between successive lines that still counts as the same table block
HEADER_SEARCH_Y_PAD = 6 # small vertical wiggle room when aligning to the header baseline

def norm(s):
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def words_to_lines(words, y_tol=LINE_Y_TOL):
    """
    Group words into visual lines by similar 'top' (y) coordinate.
    Returns a list of (y, [word dicts in order of x0])
    """
    if not words:
        return []
    # sort by y (top), then x
    words_sorted = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines = []
    cur_y = None
    cur_line = []
    for w in words_sorted:
        y = w["top"]
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur_line.append(w)
            if cur_y is None:
                cur_y = y
        else:
            if cur_line:
                # keep words in x-order
                cur_line = sorted(cur_line, key=lambda ww: ww["x0"])
                lines.append((cur_y, cur_line))
            cur_line = [w]
            cur_y = y
    if cur_line:
        cur_line = sorted(cur_line, key=lambda ww: ww["x0"])
        lines.append((cur_y, cur_line))
    return lines

def join_text(words_in_line):
    """Join a list of word dicts into a single string with spaces."""
    return norm(" ".join([w["text"] for w in words_in_line]))

def find_header_line(lines):
    """
    From a list of (y, words) lines, find the line index where the header appears
    and return (line_index, header_word_boxes_in_order).
    Matches each expected header token independently and ensures order.
    """
    for i, (_, wline) in enumerate(lines):
        texts = [norm(w["text"]).lower() for w in wline]
        # Greedy match in order: try to find each expected header token as a full word (or joined multi-word)
        matched_boxes = []
        needed = HEADER_LOWER[:]
        j = 0
        for token in needed:
            # Try exact word match first
            found_idx = None
            for k in range(j, len(wline)):
                if texts[k] == token:
                    found_idx = k
                    break
            if found_idx is None and "/" in token:
                # Some PDFs might split "Mottagare/Betalare" into three words; try to join consecutive words
                for k in range(j, len(wline)-2):
                    joined = norm(wline[k]["text"] + wline[k+1]["text"] + wline[k+2]["text"]).lower()
                    if joined == token.replace(" ", ""):
                        found_idx = k
                        # synthesize a box spanning k..k+2
                        box = {
                            "x0": wline[k]["x0"],
                            "x1": wline[k+2]["x1"],
                            "top": min(wline[k]["top"], wline[k+1]["top"], wline[k+2]["top"]),
                            "bottom": max(wline[k]["bottom"], wline[k+1]["bottom"], wline[k+2]["bottom"]),
                            "text": " ".join([wline[k]["text"], wline[k+1]["text"], wline[k+2]["text"]]),
                        }
                        matched_boxes.append(box)
                        j = k + 3
                        break
            if found_idx is not None and len(matched_boxes) < len(matched_boxes) + 1:
                # normal single-word match path
                if len(matched_boxes) < len(needed):
                    box = {
                        "x0": wline[found_idx]["x0"],
                        "x1": wline[found_idx]["x1"],
                        "top": wline[found_idx]["top"],
                        "bottom": wline[found_idx]["bottom"],
                        "text": wline[found_idx]["text"],
                    }
                    matched_boxes.append(box)
                    j = found_idx + 1

        # If we didn’t synthesize multiword for the last token and we’re short, try a softer includes-based match
        if len(matched_boxes) < 4:
            matched_boxes = []
            j = 0
            for token in needed:
                found_idx = None
                for k in range(j, len(wline)):
                    if token in texts[k]:
                        found_idx = k
                        break
                if found_idx is None:
                    matched_boxes = []
                    break
                box = {
                    "x0": wline[found_idx]["x0"],
                    "x1": wline[found_idx]["x1"],
                    "top": wline[found_idx]["top"],
                    "bottom": wline[found_idx]["bottom"],
                    "text": wline[found_idx]["text"],
                }
                matched_boxes.append(box)
                j = found_idx + 1

        if len(matched_boxes) == 4:
            return i, matched_boxes

    return None, None

def build_column_bounds(header_boxes, page_width):
    """
    Given header word boxes in order, return a list of (x_left, x_right) bounds
    for the 4 data columns. We create midpoints between header centers.
    """
    centers = [(b["x0"] + b["x1"]) / 2.0 for b in header_boxes]
    # make boundaries as: [-inf, mid(0,1)], [mid(0,1), mid(1,2)], [mid(1,2), mid(2,3)], [mid(2,3), +inf]
    mid01 = (centers[0] + centers[1]) / 2.0
    mid12 = (centers[1] + centers[2]) / 2.0
    mid23 = (centers[2] + centers[3]) / 2.0
    # add slight margins
    left_inf = -10
    right_inf = page_width + 10
    bounds = [
        (left_inf, mid01),
        (mid01,   mid12),
        (mid12,   mid23),
        (mid23,   right_inf),
    ]
    return bounds

def assign_to_column(word, col_bounds):
    xc = (word["x0"] + word["x1"]) / 2.0
    for idx, (x0, x1) in enumerate(col_bounds):
        if x0 <= xc <= x1:
            return idx
    # fallback: closest center
    centers = [ (b0 + b1)/2.0 for (b0,b1) in col_bounds ]
    diffs = [ abs(xc - c) for c in centers ]
    return min(range(len(centers)), key=lambda i: diffs[i])

def process_pdf(pdf_path: Path):
    try:
        import pdfplumber
    except ImportError:
        print("Missing dependency: pdfplumber. Install with: pip install pdfplumber")
        sys.exit(1)

    all_rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        print(f"DBG: Opened PDF: {pdf_path.name}, pages={len(pdf.pages)}")
        for pidx, page in enumerate(pdf.pages, start=1):
            print(f"\nDBG: === Page {pidx} ===")
            width = page.width

            # get all words
            try:
                words = page.extract_words(use_text_flow=True, keep_blank_chars=True)
            except Exception as e:
                print(f"DBG: extract_words failed on page {pidx}: {e}")
                continue

            if not words:
                print("DBG: No words found on this page.")
                continue

            lines = words_to_lines(words)
            print(f"DBG: Found {len(lines)} visual lines")

            header_idx, header_boxes = find_header_line(lines)
            if header_idx is None:
                print("DBG: Header not found on this page; skipping.")
                continue

            header_y = lines[header_idx][0]
            print(f"DBG: Header line index={header_idx}, y≈{header_y:.1f}")
            for name, box in zip(EXPECTED_HEADERS, header_boxes):
                print(f"DBG:  Header '{name}': x0={box['x0']:.1f}, x1={box['x1']:.1f}, text='{box['text']}'")

            col_bounds = build_column_bounds(header_boxes, page_width=width)
            for i, (x0, x1) in enumerate(col_bounds):
                print(f"DBG:  Column {i+1} bounds: [{x0:.1f}, {x1:.1f}] -> {EXPECTED_HEADERS[i]}")

            # Collect lines below header until a new header or page end
            page_rows = []
            last_line_y = None
            for li in range(header_idx + 1, len(lines)):
                y, wline = lines[li]
                # stop if we hit another header-looking line
                texts_lower = [norm(w["text"]).lower() for w in wline]
                if any(t == h for t, h in zip(texts_lower[:4], HEADER_LOWER)):
                    print(f"DBG: Encountered another header at line {li}, y≈{y:.1f}; stopping page collection.")
                    break

                # If there's a very large gap (footer area), we can heuristically stop
                if last_line_y is not None and (y - last_line_y) > 6 * ROW_GAP_ALLOW:
                    print(f"DBG: Large vertical gap after y≈{last_line_y:.1f} -> y≈{y:.1f}; likely footer; stopping.")
                    break
                last_line_y = y

                # Assign words to columns
                cols_words = [[], [], [], []]
                for w in wline:
                    col = assign_to_column(w, col_bounds)
                    cols_words[col].append(w)

                # Build cell strings (keep intra-cell word order)
                row_cells = [join_text(cw) for cw in cols_words]
                # Skip empty or spurious lines
                if all(c == "" for c in row_cells):
                    continue
                
                # --- Footer filter: skip lines with URL or page counter ---
                combined = " ".join(row_cells)
                if "http://" in combined or "https://" in combined:
                    print(f"DBG: Skipping footer-like row: {row_cells}")
                    continue
                if re.search(r"\b\d+/\d+\b", combined):  # matches "8/10"
                    print(f"DBG: Skipping page-counter row: {row_cells}")
                    continue
                # ---------------------------------------------------------

                # --- Convert date to ISO format ---
                if re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$", row_cells[0]):
                    d, m, y = row_cells[0].split(".")
                    row_cells[0] = f"{y}-{int(m):02d}-{int(d):02d}"
                # ----------------------------------

                # Simple cleanups
                # 1) Some PDFs stick the type into col 3 and the merchant into col 4;
                #    if col3 is empty but col4 starts with an uppercase token that looks like a type,
                #    try to split first token into col3
                if row_cells[2] == "" and row_cells[3] != "":
                    # Heuristic: types often look like all-caps short words at the start (e.g., KORTKÖP, ÖVERFÖRING)
                    m = re.match(r"^([A-ZÅÄÖ\-]{3,})\s+(.*)$", row_cells[3])
                    if m:
                        row_cells[2] = m.group(1)
                        row_cells[3] = m.group(2)
                page_rows.append(row_cells)

            print(f"DBG: Collected {len(page_rows)} data rows on page {pidx}")
            # Show a couple of samples
            for samp in page_rows[:3]:
                print(f"DBG:  Sample row: {samp}")

            all_rows.extend(page_rows)

    return all_rows

def write_csv(rows, out_path: Path):
    if not rows:
        print("No table rows detected. Nothing to write.")
        return
    df = pd.DataFrame(rows, columns=EXPECTED_HEADERS)
    # Drop any accidental repeated header rows
    mask_hdr = (
        df.iloc[:, 0].str.lower() == HEADER_LOWER[0]
    ) & (df.iloc[:, 1].str.lower() == HEADER_LOWER[1]) & \
        (df.iloc[:, 2].str.lower() == HEADER_LOWER[2]) & \
        (df.iloc[:, 3].str.lower() == HEADER_LOWER[3])
    df = df[~mask_hdr]
    # Strip whitespace
    df = df.applymap(norm)
    df.to_csv(out_path, sep=SEMICOLON, index=False, encoding="utf-8-sig")
    print(f"\nDBG: Wrote {len(df)} rows to {out_path.name} (semicolon-separated, UTF-8 BOM)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_pdf_table.py <input.pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    out_path = pdf_path.with_suffix(".csv")
    print(f"DBG: Starting extraction for {pdf_path.name}")
    rows = process_pdf(pdf_path)
    print(f"DBG: Total rows extracted: {len(rows)}")
    write_csv(rows, out_path)

if __name__ == "__main__":
    main()
