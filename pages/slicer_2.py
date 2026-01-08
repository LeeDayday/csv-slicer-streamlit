# app.py
# Streamlit: HWPX(2015/2022) 표 → JSON 변환기 (바로 실행 가능)
#
# 실행:
#   pip install streamlit lxml
#   streamlit run app.py

import json
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Set

import streamlit as st
from lxml import etree


# ============================================================
# Common helpers
# ============================================================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def _find_first(elem: etree._Element, want_local: str) -> Optional[etree._Element]:
    for node in elem.iter():
        if _local(node.tag) == want_local:
            return node
    return None

def _get_int(elem: Optional[etree._Element], name: str, default: int = 0) -> int:
    if elem is None:
        return default
    v = elem.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def _cell_text(tc: etree._Element) -> str:
    """
    tc 텍스트 복원:
    - <...:t> text
    - <...:nbSpace/> -> " "
    - node.tail 포함
    """
    parts: List[str] = []
    for node in tc.iter():
        ln = _local(node.tag)
        if ln == "t":
            if node.text:
                parts.append(node.text)
        elif ln == "nbSpace":
            parts.append(" ")
        if node.tail:
            parts.append(node.tail)
    return _norm("".join(parts))


# ID: [12영이03-01] 같은 대괄호 안 토큰
ID_BRACKET_PAT = re.compile(r"\[\s*([^\]]+?)\s*\]")

def split_id_and_standard(std_cell_text: str) -> Tuple[str, str]:
    t = _norm(std_cell_text)
    if not t:
        return "", ""
    m = ID_BRACKET_PAT.search(t)
    if not m:
        return "", t
    id_ = _norm(m.group(1))
    rest = _norm(t.replace(m.group(0), "", 1))
    return id_, rest

def join_sentences(parts: List[str]) -> str:
    cleaned = [_norm(p) for p in parts if _norm(p)]
    return " ".join(cleaned)


# ============================================================
# Grid builder (supports rowSpan/colSpan)
# ============================================================

@dataclass
class Cell:
    text: str
    r0: int
    c0: int
    rs: int
    cs: int

def build_grid(tbl: etree._Element) -> Tuple[List[List[Optional[Cell]]], int, int]:
    row_cnt = int(tbl.get("rowCnt") or 0)
    col_cnt = int(tbl.get("colCnt") or 0)

    tcs = tbl.xpath(".//*[local-name()='tc']")
    cells_info = []
    max_r = -1
    max_c = -1

    for tc in tcs:
        addr = _find_first(tc, "cellAddr")
        if addr is None:
            continue
        r0 = _get_int(addr, "rowAddr", 0)
        c0 = _get_int(addr, "colAddr", 0)

        span = _find_first(tc, "cellSpan")
        rs = _get_int(span, "rowSpan", 1) if span is not None else 1
        cs = _get_int(span, "colSpan", 1) if span is not None else 1

        text = _cell_text(tc)
        cells_info.append((r0, c0, rs, cs, text))

        max_r = max(max_r, r0 + rs - 1)
        max_c = max(max_c, c0 + cs - 1)

    if row_cnt <= 0:
        row_cnt = max_r + 1
    if col_cnt <= 0:
        col_cnt = max_c + 1

    if row_cnt <= 0 or col_cnt <= 0:
        return [], 0, 0

    grid: List[List[Optional[Cell]]] = [[None for _ in range(col_cnt)] for _ in range(row_cnt)]

    for r0, c0, rs, cs, text in cells_info:
        cell = Cell(text=text, r0=r0, c0=c0, rs=rs, cs=cs)
        for r in range(r0, min(row_cnt, r0 + rs)):
            for c in range(c0, min(col_cnt, c0 + cs)):
                grid[r][c] = cell

    return grid, row_cnt, col_cnt

def row_unique_texts(grid: List[List[Optional[Cell]]], r: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for c, cell in enumerate(grid[r]):
        if cell is None:
            out.append("")
            continue
        key = (cell.r0, cell.c0)
        if key in seen:
            out.append("")
        else:
            seen.add(key)
            out.append(cell.text)
    return out


# ============================================================
# 2015 parser (상/중/하) → JSON with (A)(B)(C) + order preserved
# ============================================================

LEVELS_2015 = {"상": "high", "중": "mid", "하": "low"}

# 성취기준 셀 내부의 "[평가준거 성취기준 ...]" 표기 제거(결과엔 포함 안함)
CRIT_MARK_PAT = re.compile(r"\[\s*평가준거\s*성취기준.*?\]", re.UNICODE)

def clean_standard_text_2015(s: str) -> str:
    s = _norm(s)
    if not s:
        return ""
    s = CRIT_MARK_PAT.sub("", s)
    return _norm(s)

def extract_level_2015(text: str) -> str:
    # 공백/줄바꿈 섞여도 상/중/하 추출
    if not text:
        return ""
    t = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = _norm(t)
    if len(t) <= 10:
        m = re.search(r"(상|중|하)", t)
        if m:
            return m.group(1)
    return ""

def strip_level_prefix_2015(s: str) -> str:
    # 평가 문장 앞에 "상 중 하" 같은 토큰이 섞인 경우 제거
    if not s:
        return ""
    t = s.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = _norm(t)
    for _ in range(3):
        t2 = re.sub(r"^(상|중|하)\s+", "", t)
        if t2 == t:
            break
        t = t2
    return t

@dataclass
class ColMap2015:
    std_col: int
    eval_col: int
    level_col: int
    eval_text_cols: List[int]
    header_r: int

def detect_colmap_2015(grid: List[List[Optional[Cell]]], row_cnt: int, col_cnt: int) -> Optional[ColMap2015]:
    if row_cnt == 0 or col_cnt == 0:
        return None

    header_r = 0
    for r in range(min(3, row_cnt)):
        txts = " ".join(row_unique_texts(grid, r))
        if ("교육과정" in txts and "성취기준" in txts) and ("평가" in txts and "기준" in txts):
            header_r = r
            break

    header = [_norm(x) for x in row_unique_texts(grid, header_r)]

    def find_idx(pred) -> Optional[int]:
        for i, t in enumerate(header):
            if pred(t):
                return i
        return None

    std_i = find_idx(lambda t: "교육과정" in t and "성취기준" in t)
    eval_i = find_idx(lambda t: "평가" in t and "기준" in t)
    if std_i is None or eval_i is None:
        return None

    # 레벨 컬럼 찾기(정확 일치 X, extract_level 사용)
    level_col = None
    for r in range(header_r + 1, min(row_cnt, header_r + 6)):
        for c in range(eval_i, col_cnt):
            cell = grid[r][c]
            if cell is None:
                continue
            if extract_level_2015(cell.text) in LEVELS_2015:
                level_col = c
                break
        if level_col is not None:
            break
    if level_col is None:
        level_col = eval_i

    eval_text_cols = [c for c in range(level_col + 1, col_cnt)]
    return ColMap2015(std_col=std_i, eval_col=eval_i, level_col=level_col, eval_text_cols=eval_text_cols, header_r=header_r)

def parse_table_2015_to_setrows(tbl: etree._Element, set_id_start: int) -> Tuple[List[Dict[str, Any]], int]:
    grid, row_cnt, col_cnt = build_grid(tbl)
    if row_cnt <= 1 or col_cnt <= 1:
        return [], set_id_start

    cmap = detect_colmap_2015(grid, row_cnt, col_cnt)
    if cmap is None:
        return [], set_id_start

    start_r = cmap.header_r + 1
    out_rows: List[Dict[str, Any]] = []

    # 한 평가세트에 매핑될 성취기준들(= n:1 처리용)
    stds_in_block: List[str] = []
    last_std_raw = ""
    eval_buf = {"high": "", "mid": "", "low": ""}

    set_id = set_id_start

    def add_std(std_raw: str):
        nonlocal stds_in_block
        std_raw = clean_standard_text_2015(std_raw)
        if not std_raw:
            return
        if std_raw not in stds_in_block:
            stds_in_block.append(std_raw)

    def reset_block():
        nonlocal eval_buf, stds_in_block
        eval_buf = {"high": "", "mid": "", "low": ""}
        stds_in_block = []

    def flush_if_complete():
        nonlocal eval_buf, stds_in_block, set_id
        if eval_buf["high"] and eval_buf["mid"] and eval_buf["low"]:
            targets = stds_in_block[:] if stds_in_block else ([last_std_raw] if last_std_raw else [])
            for std_raw in targets:
                id_, standard = split_id_and_standard(std_raw)
                out_rows.append({
                    "_set_id": set_id,
                    "id": id_,
                    "standard": standard,
                    "high": eval_buf["high"],
                    "mid": eval_buf["mid"],
                    "low": eval_buf["low"],
                })
            set_id += 1
            reset_block()

    for r in range(start_r, row_cnt):
        # 성취기준 갱신(셀의 top-left에서만)
        sc = grid[r][cmap.std_col]
        std_raw = ""
        if sc is not None and sc.r0 == r and sc.c0 == cmap.std_col:
            std_raw = sc.text

        std_raw = clean_standard_text_2015(std_raw)
        if std_raw:
            last_std_raw = std_raw
            add_std(last_std_raw)

        # 레벨
        lc = grid[r][cmap.level_col]
        lvl = extract_level_2015(lc.text) if lc is not None else ""
        if lvl not in LEVELS_2015:
            continue
        key = LEVELS_2015[lvl]

        # 평가 텍스트(중복 셀 제거)
        parts: List[str] = []
        used = set()
        for c in cmap.eval_text_cols:
            ec = grid[r][c]
            if ec is None:
                continue
            ck = (ec.r0, ec.c0)
            if ck in used:
                continue
            used.add(ck)
            t = _norm(ec.text)
            if t:
                parts.append(t)

        eval_text = strip_level_prefix_2015(_norm(" ".join(parts)))

        # 새 high 시작 시 세트 경계 처리
        if key == "high":
            if eval_buf["high"] or eval_buf["mid"] or eval_buf["low"]:
                if eval_buf["high"] and eval_buf["mid"] and eval_buf["low"]:
                    flush_if_complete()
                else:
                    reset_block()
                    if last_std_raw:
                        add_std(last_std_raw)

        eval_buf[key] = eval_text
        flush_if_complete()

    return out_rows, set_id

def build_json_2015_from_setrows(setrows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (A)(B)(C) + 표 순서 유지:
      - 같은 _set_id에 id가 2개 이상이면 (B) n:1
      - 같은 id가 여러 _set_id에 나타나면 (C) 1:n로 evaluation 이어붙임
      - 최종 결과는 first_set_id(최초 등장 set_id)로 정렬
    """
    by_set: Dict[int, List[Dict[str, Any]]] = {}
    for r in setrows:
        by_set.setdefault(r["_set_id"], []).append(r)

    set_ids = sorted(by_set.keys())

    temp_items: List[Dict[str, Any]] = []
    for sid in set_ids:
        g = by_set[sid]
        ids = [x["id"] for x in g if x.get("id")]
        standards = [x["standard"] for x in g if x.get("standard")]

        unique_ids: List[str] = []
        for x in ids:
            if x not in unique_ids:
                unique_ids.append(x)

        e = {"high": g[0]["high"], "mid": g[0]["mid"], "low": g[0]["low"]}

        if len(unique_ids) >= 2:
            # (B) n:1
            temp_items.append({
                "_kind": "B",
                "first_set_id": sid,
                "id": unique_ids,
                "standard": join_sentences(standards),
                "evaluation": e,
            })
        else:
            # (A) or (C) part
            one_id = unique_ids[0] if unique_ids else (g[0].get("id") or "")
            one_std = g[0].get("standard") or ""
            temp_items.append({
                "_kind": "S",
                "first_set_id": sid,
                "id": one_id,
                "standard": one_std,
                "evaluation": e,
            })

    merged_list: List[Dict[str, Any]] = []
    merged_by_id: Dict[str, Dict[str, Any]] = {}

    for item in temp_items:
        if item["_kind"] == "B":
            merged_list.append(item)
            continue

        id_ = item.get("id") or ""
        if not id_:
            merged_list.append(item)
            continue

        if id_ not in merged_by_id:
            merged_by_id[id_] = {
                "_kind": "S",
                "first_set_id": item["first_set_id"],
                "id": id_,
                "standard": item.get("standard") or "",
                "_parts": {
                    "high": [item["evaluation"]["high"]],
                    "mid":  [item["evaluation"]["mid"]],
                    "low":  [item["evaluation"]["low"]],
                }
            }
        else:
            merged_by_id[id_]["_parts"]["high"].append(item["evaluation"]["high"])
            merged_by_id[id_]["_parts"]["mid"].append(item["evaluation"]["mid"])
            merged_by_id[id_]["_parts"]["low"].append(item["evaluation"]["low"])

    for v in merged_by_id.values():
        merged_list.append({
            "_kind": "S",
            "first_set_id": v["first_set_id"],
            "id": v["id"],
            "standard": v["standard"],
            "evaluation": {
                "high": join_sentences(v["_parts"]["high"]),
                "mid":  join_sentences(v["_parts"]["mid"]),
                "low":  join_sentences(v["_parts"]["low"]),
            }
        })

    merged_list.sort(key=lambda x: x["first_set_id"])

    out: List[Dict[str, Any]] = []
    for x in merged_list:
        out.append({
            "id": x["id"],
            "standard": x["standard"],
            "evaluation": x["evaluation"],
        })
    return out

def parse_hwpx_2015_to_result(hwpx_bytes: bytes) -> Dict[str, Any]:
    all_setrows: List[Dict[str, Any]] = []
    per_table: List[Dict[str, Any]] = []

    with zipfile.ZipFile(BytesIO(hwpx_bytes), "r") as z:
        names = z.namelist()
        section_files = sorted([n for n in names if re.search(r"Contents/section\d+\.xml$", n)])
        if not section_files:
            section_files = sorted([n for n in names if n.lower().endswith(".xml") and "section" in n.lower()])

        parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=False)

        set_id = 0
        table_index = 0
        for sf in section_files:
            root = etree.fromstring(z.read(sf), parser=parser)
            tbls = root.xpath("//*[local-name()='tbl' or local-name()='table']")
            for tbl in tbls:
                rows, set_id = parse_table_2015_to_setrows(tbl, set_id)
                if rows:
                    all_setrows.extend(rows)
                    per_table.append({
                        "table_index": table_index,
                        "items": build_json_2015_from_setrows(rows),
                    })
                table_index += 1

    items = build_json_2015_from_setrows(all_setrows)
    return {"table_count": len(per_table), "tables": per_table, "items": items}


# ============================================================
# 2022 parser (A~E OR A~C 등 부분집합) → JSON items
# ============================================================


LEVELS_2022 = {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}
ORDER_2022 = ["a", "b", "c", "d", "e"]

def detect_2022_header_row(grid: List[List[Optional[Cell]]], row_cnt: int) -> Optional[int]:
    for r in range(min(3, row_cnt)):
        txts = " ".join(_norm(x) for x in row_unique_texts(grid, r))
        if ("성취기준" in txts) and ("성취수준" in txts):
            return r
    return None

def parse_table_2022(tbl: etree._Element) -> List[Dict[str, Any]]:
    """
    2022 표 지원:
    - 성취수준이 A~E가 아니라 A~C만 있는 표도 존재
    - 레벨(A/B/...) 칸이 rowSpan으로 병합되어 아래 행은 빈 문자열인 경우가 있음 -> 직전 레벨로 처리
    - 가장 중요한 수정: 레벨을 다 모으기 전에는 flush하지 않고,
      '다음 성취기준 시작' 또는 '테이블 끝'에서만 flush한다.
    - 출력 evaluation은 "실제로 등장한 최대 레벨"까지만 키 포함 (예: a~c)
      (d/e가 없으면 키 자체를 출력하지 않음)
    """
    grid, row_cnt, col_cnt = build_grid(tbl)
    if row_cnt <= 1 or col_cnt < 3:
        return []

    header_r = detect_2022_header_row(grid, row_cnt)
    if header_r is None:
        return []

    # 일반적으로:
    # col0: 성취기준([id] + 문장)
    # col1: A/B/C/D/E
    # col2: 설명 텍스트
    std_col, level_col, text_col = 0, 1, 2

    out: List[Dict[str, Any]] = []

    current_std_raw: str = ""
    buf: Dict[str, List[str]] = {k: [] for k in ORDER_2022}
    seen_levels: List[str] = []  # 등장 순서(최대 레벨 판단용)
    last_level_key: Optional[str] = None

    def finalize_current():
        nonlocal current_std_raw, buf, seen_levels, last_level_key

        if not current_std_raw:
            return

        # 어떤 레벨이든 최소 1개는 있어야 의미 있음
        present = [k for k in ORDER_2022 if buf[k]]
        if not present:
            current_std_raw = ""
            buf = {k: [] for k in ORDER_2022}
            seen_levels = []
            last_level_key = None
            return

        # 등장한 최대 레벨까지만 출력 키 포함 (예: a~c)
        max_i = max(ORDER_2022.index(k) for k in present)
        keys_to_emit = ORDER_2022[: max_i + 1]

        id_, standard = split_id_and_standard(current_std_raw)

        out.append({
            "id": id_,
            "standard": standard,
            "evaluation": {k: join_sentences(buf[k]) for k in keys_to_emit}
        })

        # reset
        current_std_raw = ""
        buf = {k: [] for k in ORDER_2022}
        seen_levels = []
        last_level_key = None

    # 본문 파싱
    for r in range(header_r + 1, row_cnt):
        # 새 성취기준 시작 감지 (std_col에서 top-left일 때)
        std_cell = grid[r][std_col]
        if std_cell is not None and std_cell.r0 == r and std_cell.c0 == std_col:
            # 이전 성취기준 마무리
            finalize_current()

            current_std_raw = _norm(std_cell.text)
            buf = {k: [] for k in ORDER_2022}
            seen_levels = []
            last_level_key = None

        if not current_std_raw:
            continue

        # 레벨 읽기
        lvl_cell = grid[r][level_col]
        lvl_txt = _norm(lvl_cell.text) if lvl_cell is not None else ""

        # 레벨이 있으면 업데이트, 없으면 직전 레벨 사용(rowSpan 대응)
        if lvl_txt in LEVELS_2022:
            last_level_key = LEVELS_2022[lvl_txt]
            seen_levels.append(last_level_key)

        if last_level_key is None:
            # 레벨도 없고 직전 레벨도 없으면 매핑 불가
            continue

        # 설명 텍스트 읽기
        txt_cell = grid[r][text_col]
        val = _norm(txt_cell.text) if txt_cell is not None else ""
        if not val:
            continue

        buf[last_level_key].append(val)

    # 테이블 끝에서 마지막 성취기준 마무리
    finalize_current()

    return out

def parse_hwpx_2022_to_result(hwpx_bytes: bytes) -> Dict[str, Any]:
    all_items: List[Dict[str, Any]] = []
    per_table: List[Dict[str, Any]] = []

    with zipfile.ZipFile(BytesIO(hwpx_bytes), "r") as z:
        names = z.namelist()
        section_files = sorted([n for n in names if re.search(r"Contents/section\d+\.xml$", n)])
        if not section_files:
            section_files = sorted([n for n in names if n.lower().endswith(".xml") and "section" in n.lower()])

        parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=False)

        table_index = 0
        for sf in section_files:
            root = etree.fromstring(z.read(sf), parser=parser)
            tbls = root.xpath("//*[local-name()='tbl' or local-name()='table']")
            for tbl in tbls:
                items = parse_table_2022(tbl)
                if items:
                    all_items.extend(items)
                    per_table.append({"table_index": table_index, "items": items})
                table_index += 1

    return {"table_count": len(per_table), "tables": per_table, "items": all_items}


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="HWPX Table → JSON Parser", layout="wide")
st.title("HWPX 표 → JSON 변환기 (2015/2022)")

uploaded = st.file_uploader("HWPX 파일 업로드", type=["hwpx"])

if uploaded is None:
    st.info("`.hwpx` 파일을 업로드하면 표를 추출해 규칙에 맞게 JSON으로 변환합니다.")
    st.stop()

hwpx_bytes = uploaded.read()

edition = st.radio(
    "평가기준 버전",
    ["2015 개정 (상/중/하)", "2022 개정 (A~E 또는 A~C 등)"],
    horizontal=True
)

with st.spinner("표 추출 및 파싱 중..."):
    if edition.startswith("2015"):
        result = parse_hwpx_2015_to_result(hwpx_bytes)
    else:
        result = parse_hwpx_2022_to_result(hwpx_bytes)

st.success(f"완료! 추출된 표 개수: {result['table_count']}")

view_mode = st.radio("표시 범위", ["전체 items만 보기", "테이블별 보기"], horizontal=True)

payload = result["items"] if view_mode == "전체 items만 보기" else result["tables"]
json_text = json.dumps(payload, ensure_ascii=False, indent=2)

st.subheader("JSON 미리보기")
st.code(json_text, language="json")

st.subheader("바로 복사하기")
st.text_area("아래 내용을 선택해서 복사하거나, 버튼으로 복사하세요.", json_text, height=220)

st.components.v1.html(
    f"""
    <button style="padding:10px 14px; font-size:14px; cursor:pointer;"
            onclick="navigator.clipboard.writeText({json.dumps(json_text)}).then(()=>alert('클립보드에 복사했어요!'));">
      JSON 복사
    </button>
    """,
    height=60
)

st.subheader("JSON 다운로드")
download_name = uploaded.name.rsplit(".", 1)[0] + (".2015.json" if edition.startswith("2015") else ".2022.json")
st.download_button(
    label="JSON 파일 다운로드",
    data=json_text.encode("utf-8"),
    file_name=download_name,
    mime="application/json",
)

st.divider()
st.subheader("검증")

if edition.startswith("2015"):
    bad = [x for x in result["items"] if isinstance(x.get("id"), str) and len(x["id"].split("-")) >= 4]
    st.write(f"자식(평가준거) id로 보이는 항목 개수: **{len(bad)}** (0에 가까울수록 좋음)")
else:
    # 2022는 A~E가 모두 안 나올 수 있으니, "evaluation이 비어있는 항목"만 체크
    empty_eval = [x for x in result["items"] if not any(_norm(v) for v in x.get("evaluation", {}).values())]
    st.write(f"evaluation이 전부 빈 항목 개수: **{len(empty_eval)}** (0이어야 정상)")
