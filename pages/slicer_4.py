# app.py
import re
import json
import zipfile
import io
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, OrderedDict

import streamlit as st


# =========================
# Utils
# =========================
NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}

def _ns(tag: str, nsmap: Dict[str, str]) -> str:
    prefix, local = tag.split(":")
    return f"{{{nsmap[prefix]}}}{local}"

def _normalize_header(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()

def _normalize_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s).strip()
    return s

def _clean_inline(s: str) -> str:
    s = _normalize_text(s).replace("\n", " ").strip()
    s = s.replace("", "·")
    s = s.replace("ㆍ", "･")
    s = re.sub(r"\s+", " ", s)
    return s

def _is_table_caption(s: str) -> bool:
    s = _clean_inline(s)
    return s.startswith("<표")

def to_pretty_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def split_eval_elements(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    if any(b in t for b in ["◦", "○", "•", "∙"]):
        parts = re.split(r"(?:^|\n)\s*[◦○•∙]\s*", "\n" + t)
        return [_clean_inline(p) for p in parts if p.strip()]
    return [_clean_inline(p) for p in t.split("\n") if p.strip()]


# =========================
# XML extraction helpers
# =========================
NSMAP = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
HP_TBL = _ns("hp:tbl", NSMAP)
HP_TR  = _ns("hp:tr", NSMAP)
HP_TC  = _ns("hp:tc", NSMAP)
HP_T   = _ns("hp:t", NSMAP)
HP_P   = _ns("hp:p", NSMAP)

def extract_all_hp_text(elem: ET.Element) -> List[str]:
    out: List[str] = []
    for t in elem.iter(HP_T):
        txt = "".join(t.itertext()).strip()
        if txt:
            out.append(txt)
    return out

def extract_cell_text(tc: ET.Element) -> str:
    parts = extract_all_hp_text(tc)
    return _normalize_text(" ".join(parts))

def table_to_matrix(tbl: ET.Element) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in tbl.iter(HP_TR):
        row = []
        for tc in tr.findall(HP_TC):
            row.append(extract_cell_text(tc))
        if row:
            rows.append(row)
    return rows

def iter_outer_paragraphs(root: ET.Element):
    parent = {}
    for par in root.iter():
        for ch in list(par):
            parent[ch] = par

    def has_ancestor(elem: ET.Element, tags: set) -> bool:
        cur = elem
        while cur in parent:
            cur = parent[cur]
            if cur.tag in tags:
                return True
        return False

    for p in root.iter(HP_P):
        if has_ancestor(p, {HP_TBL, HP_TC}):
            continue
        yield p


# =========================
# HWPX loader
# =========================
_SECTION_RE = re.compile(r"^Contents/section(\d+)\.xml$")

def _section_sort_key(name: str) -> Tuple[int, str]:
    m = _SECTION_RE.match(name)
    return (int(m.group(1)), name) if m else (10**9, name)

def read_hwpx_sections_from_bytes(hwpx_bytes: bytes) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(hwpx_bytes), "r") as z:
        section_names = [n for n in z.namelist() if _SECTION_RE.match(n)]
        section_names.sort(key=_section_sort_key)

        out = []
        for name in section_names:
            xml_text = z.read(name).decode("utf-8", errors="replace")
            out.append((name, xml_text))
        return out


# =========================
# 2022: grid builder (merge-aware)
# =========================
def _cell_text(tc: ET.Element) -> str:
    parts: List[str] = []
    for node in tc.iter():
        tag = node.tag
        if tag == f"{{{NS['hp']}}}t":
            if node.text:
                parts.append(node.text)
        elif tag == f"{{{NS['hp']}}}lineBreak":
            parts.append("\n")
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_grid_from_tbl(tbl: ET.Element) -> List[List[str]]:
    row_cnt = int(tbl.attrib.get("rowCnt", "0") or "0")
    col_cnt = int(tbl.attrib.get("colCnt", "0") or "0")

    if row_cnt == 0 or col_cnt == 0:
        max_r = -1
        max_c = -1
        for tc in tbl.findall(".//hp:tc", NS):
            addr = tc.find("./hp:cellAddr", NS)
            span = tc.find("./hp:cellSpan", NS)
            if addr is None:
                continue
            c0 = int(addr.attrib.get("colAddr", "0"))
            r0 = int(addr.attrib.get("rowAddr", "0"))
            cs = int(span.attrib.get("colSpan", "1")) if span is not None else 1
            rs = int(span.attrib.get("rowSpan", "1")) if span is not None else 1
            max_r = max(max_r, r0 + rs - 1)
            max_c = max(max_c, c0 + cs - 1)
        row_cnt = max_r + 1
        col_cnt = max_c + 1

    if row_cnt <= 0 or col_cnt <= 0:
        return []

    grid = [["" for _ in range(col_cnt)] for _ in range(row_cnt)]

    for tc in tbl.findall(".//hp:tc", NS):
        addr = tc.find("./hp:cellAddr", NS)
        span = tc.find("./hp:cellSpan", NS)
        if addr is None:
            continue

        c0 = int(addr.attrib.get("colAddr", "0"))
        r0 = int(addr.attrib.get("rowAddr", "0"))
        cs = int(span.attrib.get("colSpan", "1")) if span is not None else 1
        rs = int(span.attrib.get("rowSpan", "1")) if span is not None else 1

        txt = _cell_text(tc)
        if not txt:
            continue

        for r in range(r0, min(row_cnt, r0 + rs)):
            for c in range(c0, min(col_cnt, c0 + cs)):
                if not grid[r][c]:
                    grid[r][c] = txt

    return grid


# =========================
# 2022: section/title detection
# =========================
RE_MAJOR = re.compile(r"^\s*([123])\s*(성취기준별\s*성취수준|영역별\s*성취수준|예시\s*평가\s*도구)\b")
RE_SUBTITLE = re.compile(r"^\(\d+\)\s*.+")      # (숫자) 문장

def table_text_flat(tbl: ET.Element) -> str:
    m = table_to_matrix(tbl)
    flat = " ".join(_clean_inline(c) for r in m for c in r if c)
    return re.sub(r"\s+", " ", flat).strip()

def detect_major_by_title_table(tbl: ET.Element) -> Optional[str]:
    """
    ✅ "진짜 목차 타이틀 표"만 major로 인정
    - 1행 2열
    - left가 1/2/3
    - right가 해당 제목
    """
    if tbl.get("rowCnt") != "1" or tbl.get("colCnt") != "2":
        return None

    m = table_to_matrix(tbl)
    if not m or len(m[0]) < 2:
        return None

    left = _clean_inline(m[0][0])
    right = _clean_inline(m[0][1])
    left_num = re.sub(r"\D", "", left)

    right_norm = right.replace(" ", "")

    if left_num == "1" and ("성취기준별" in right_norm and "성취수준" in right_norm):
        return "1"
    if left_num == "2" and ("영역별" in right_norm and "성취수준" in right_norm):
        return "2"
    if left_num == "3" and ("예시" in right_norm and "평가" in right_norm and "도구" in right_norm):
        return "3"
    return None


# =========================
# 2022: Subject detection
# =========================
SUBJECT_LINE_RE = re.compile(r"^\s*(.+?)\s*성취수준\s*$")

def detect_subject_line(para_text: str) -> Optional[str]:
    t = _clean_inline(para_text)
    if not t:
        return None

    ban = ["2022", "개정 교육과정", "교육과정에 따른"]
    if any(b in t for b in ban):
        return None

    m = SUBJECT_LINE_RE.match(t)
    if not m:
        return None

    subject = _clean_inline(m.group(1))
    if len(subject) < 2:
        return None

    subject = subject.strip("ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ ").strip()
    if not subject or subject in {"(영역)", "영역"}:
        return None

    return subject


# =========================
# 2022: Achievement parsing
# =========================
LEVEL_RE = re.compile(r"^[A-E]$|^P$")

def _is_level(s: str) -> bool:
    return bool(LEVEL_RE.match(_clean_inline(s)))

def _norm_category(s: str) -> Optional[str]:
    s = _clean_inline(s).replace("·", "･")
    if "지식" in s:
        return "지식･이해"
    if "과정" in s:
        return "과정･기능"
    if "가치" in s:
        return "가치･태도"
    return None

def _find_2022_ach_header_row(matrix: List[List[str]], scan_rows: int = 12) -> Optional[int]:
    key0 = _normalize_header("영역")
    key1 = _normalize_header("영역별 성취수준")

    for r_idx in range(min(scan_rows, len(matrix))):
        row = matrix[r_idx]
        tokens = {_normalize_header(c) for c in row if _normalize_header(c)}
        if key0 in tokens and key1 in tokens:
            return r_idx

        row_blob = _normalize_header(" ".join(c for c in row if c))
        if key0 in row_blob and key1 in row_blob:
            return r_idx
    return None

def _should_skip_by_level_guide(ctx: List[str], matrix: List[List[str]]) -> bool:
    blob = " ".join(_clean_inline(x) for x in ctx[-10:])
    blob += " " + " ".join(_clean_inline(c) for r in matrix[:3] for c in r if c)
    norm = blob.replace(" ", "")
    if ("5수준구분" in norm) or ("3수준구분" in norm):
        return True
    if ("수준구분" in norm and ("5" in norm or "3" in norm)):
        return True
    return False

def parse_ach_grid(grid: List[List[str]]) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], str]:
    if not grid:
        return {}, ""

    header_row_idx = _find_2022_ach_header_row(grid, scan_rows=min(20, len(grid)))
    if header_row_idx is None:
        return {}, ""

    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    cur_area = ""
    cur_level = ""
    first_area = ""

    for r in range(header_row_idx + 1, len(grid)):
        row = [(_clean_inline(x) if x else "") for x in grid[r]]
        if not any(row):
            continue
        if len(row) < 4:
            row += [""] * (4 - len(row))

        area = row[0] or cur_area
        level = row[1] or cur_level
        category = row[2] if len(row) > 2 else ""
        desc = " ".join(x for x in row[3:] if x).strip()

        if area:
            cur_area = area
            if not first_area:
                first_area = area
        if level:
            cur_level = level

        if not _is_level(cur_level):
            continue

        cat = _norm_category(category) if category else None
        if not cat or not desc:
            continue

        out.setdefault(cur_area, {}).setdefault(cur_level, {})[cat] = _clean_inline(desc)

    return out, first_area

def parse_achievement_table_2022(tbl: ET.Element, ctx: List[str]) -> Optional[Dict[str, Any]]:
    matrix0 = table_to_matrix(tbl)
    if matrix0 and _should_skip_by_level_guide(ctx, matrix0):
        return None

    grid = build_grid_from_tbl(tbl)
    parsed, first_area = parse_ach_grid(grid)
    if not parsed:
        return None

    items: List[Dict[str, str]] = []
    for area, lv_map in parsed.items():
        for level, cat_map in lv_map.items():
            for category, desc in cat_map.items():
                items.append({
                    "level": level,
                    "category": category,
                    "description": desc,
                })

    if not items:
        return None

    return {
        "_first_area": first_area,
        "title": "2 영역별 성취수준",
        "subtitle": "",
        "achievement_levels": items
    }


# =========================
# 2022: Assessment parsing
# =========================
CAN_ITEM = _normalize_header("문항번호")
CAN_UNIT = _normalize_header("영역/단원명")
CAN_CODE = _normalize_header("교육과정성취기준코드")
CAN_TYPE = _normalize_header("문항유형")
CAN_ELEM = _normalize_header("평가요소")

ASSESS_HEADER_ALIASES = {
    _normalize_header("문항번호"): CAN_ITEM,
    _normalize_header("문항 번호"): CAN_ITEM,
    _normalize_header("영역/단원명"): CAN_UNIT,
    _normalize_header("영역"): CAN_UNIT,
    _normalize_header("교육과정성취기준코드"): CAN_CODE,
    _normalize_header("교육과정 성취기준 코드"): CAN_CODE,
    _normalize_header("평가준거성취기준코드"): CAN_CODE,
    _normalize_header("평가준거 성취기준 코드"): CAN_CODE,
    _normalize_header("문항유형"): CAN_TYPE,
    _normalize_header("문항 유형"): CAN_TYPE,
    _normalize_header("평가요소"): CAN_ELEM,
    _normalize_header("평가 요소"): CAN_ELEM,

    _normalize_header("번호"): CAN_ITEM,
    _normalize_header("성취기준"): CAN_CODE,
    _normalize_header("평가 도구 유형"): CAN_TYPE,
    _normalize_header("평가도구유형"): CAN_TYPE,
    _normalize_header("평가 요소/ 평가 과제"): CAN_ELEM,
    _normalize_header("평가요소/평가과제"): CAN_ELEM,
    _normalize_header("평가요소/ 평가과제"): CAN_ELEM,
    _normalize_header("평가 요소/평가 과제"): CAN_ELEM,
}

REQUIRED_ASSESS_CANON = {CAN_ITEM, CAN_UNIT, CAN_CODE, CAN_TYPE, CAN_ELEM}

def parse_assessment_table(tbl: ET.Element, extra_aliases: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, Any]]]:
    matrix = table_to_matrix(tbl)
    if not matrix or len(matrix) < 2:
        return None

    aliases = dict(ASSESS_HEADER_ALIASES)
    if extra_aliases:
        for k, v in extra_aliases.items():
            if k and v:
                aliases[_normalize_header(k)] = _normalize_header(v)

    def canon(h: str) -> str:
        nh = _normalize_header(h)
        return aliases.get(nh, nh)

    header_row_idx = None
    for r_idx in range(min(10, len(matrix))):
        header_canon = [canon(h) for h in matrix[r_idx]]
        if REQUIRED_ASSESS_CANON.issubset(set(header_canon)):
            header_row_idx = r_idx
            break
    if header_row_idx is None:
        return None

    header_canon = [canon(h) for h in matrix[header_row_idx]]

    def idx(name: str) -> int:
        return header_canon.index(name)

    col_item = idx(CAN_ITEM)
    col_unit = idx(CAN_UNIT)
    col_code = idx(CAN_CODE)
    col_type = idx(CAN_TYPE)
    col_elem = idx(CAN_ELEM)

    items: List[Dict[str, Any]] = []
    for r in matrix[header_row_idx + 1:]:
        if len(r) <= max(col_item, col_unit, col_code, col_type, col_elem):
            continue

        item_number = _clean_inline(r[col_item])
        domain_unit = _clean_inline(r[col_unit])
        curriculum_code = _clean_inline(r[col_code])
        assessment_type = _clean_inline(r[col_type])
        eval_text = _normalize_text(r[col_elem])

        if not any([item_number, domain_unit, curriculum_code, assessment_type, eval_text]):
            continue

        items.append({
            "item_number": item_number,
            "domain_unit": domain_unit,
            "curriculum_code": curriculum_code,
            "assessment_type": assessment_type,
            "evaluation_elements": split_eval_elements(eval_text),
        })

    return items if items else None


# =========================
# Main parse_2022
# =========================
def parse_2022(hwpx_bytes: bytes, extra_assess_aliases: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    sections = read_hwpx_sections_from_bytes(hwpx_bytes)

    current_major: Optional[str] = None   # "1" | "2" | "3"
    current_subject: str = "UNKNOWN"
    pending_subject: Optional[str] = None

    last_subtitle: str = ""  # (숫자) 문장 (2 구간) 또는 고정문구(3 구간)

    recent_paras = deque(maxlen=30)

    for sec_idx, (_, xml_text) in enumerate(sections):
        root = ET.fromstring(xml_text)
        para_no = -1

        for p in iter_outer_paragraphs(root):
            para_no += 1
            tbl = p.find(".//" + HP_TBL)

            # ---------- TABLE ----------
            if tbl is not None:
                # ✅ 핵심 패치: major 판정은 "진짜 1행2열 타이틀 표"만 사용
                major = detect_major_by_title_table(tbl)
                if major:
                    current_major = major
                    recent_paras.clear()

                    if major == "1":
                        if pending_subject:
                            current_subject = pending_subject
                            pending_subject = None
                        last_subtitle = ""

                    elif major == "2":
                        last_subtitle = ""

                    elif major == "3":
                        last_subtitle = "가. 예시 평가 도구 개요"

                    continue

                # ✅ 2 영역별 성취수준: ach table 파싱
                if current_major == "2":
                    ach_obj = parse_achievement_table_2022(tbl, ctx=list(recent_paras))
                    if ach_obj is not None:
                        ach_obj["title"] = "2 영역별 성취수준"
                        ach_obj["subtitle"] = last_subtitle  # "(n) 문장" 그대로

                        ach_obj["_subject"] = current_subject
                        ach_obj["_order"] = (sec_idx, para_no)

                        results.append(ach_obj)
                        continue

                # ✅ 3 예시 평가 도구: ass table 파싱
                if current_major == "3":
                    ass_items = parse_assessment_table(tbl, extra_aliases=extra_assess_aliases)
                    if ass_items is not None:
                        results.append({
                            "_subject": current_subject,
                            "_order": (sec_idx, para_no),
                            "title": "3 예시 평가 도구",
                            "subtitle": last_subtitle,
                            "assessment_items": ass_items
                        })
                        continue

                continue

            # ---------- PARAGRAPH ----------
            texts = extract_all_hp_text(p)
            if not texts:
                continue
            para_text = _clean_inline(" ".join(texts))
            if not para_text or _is_table_caption(para_text):
                continue

            subj = detect_subject_line(para_text)
            if subj:
                pending_subject = subj
                recent_paras.append(para_text)
                continue

            m = RE_MAJOR.match(para_text)
            if m:
                current_major = m.group(1)
                recent_paras.clear()
                if current_major == "2":
                    last_subtitle = ""
                elif current_major == "3":
                    last_subtitle = "가. 예시 평가 도구 개요"
                else:
                    last_subtitle = ""
                recent_paras.append(para_text)
                continue

            if current_major == "2" and RE_SUBTITLE.match(para_text):
                last_subtitle = para_text
                recent_paras.append(para_text)
                continue

            recent_paras.append(para_text)

    results.sort(key=lambda x: x.get("_order", (10**9, 10**9)))
    for r in results:
        r.pop("_order", None)
        r.pop("_first_area", None)
    return results


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="HWPX 파서 (2015/2022)", layout="wide")
st.title("HWPX > JSON 파서")

st.markdown(
    """
    <style>
    div[data-testid="stCodeBlock"] pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- session state init ----------
if "parsed_grouped_by_file" not in st.session_state:
    st.session_state.parsed_grouped_by_file = {}
if "parsed_results_by_file" not in st.session_state:
    st.session_state.parsed_results_by_file = {}
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "subject_query" not in st.session_state:
    st.session_state.subject_query = ""
if "show_all_subjects" not in st.session_state:
    st.session_state.show_all_subjects = False
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None

mode = st.sidebar.radio("Mode", ["2022", "2015 (placeholder)"], index=0)
uploaded = st.file_uploader("파일 업로드 (.hwpx)", type=["hwpx"], accept_multiple_files=True)

st.sidebar.header("Assessment header overrides (optional)")
extra_assess_aliases_raw = st.sidebar.text_area(
    "Extra ASSESS header aliases (JSON dict)",
    value="{}",
    height=140,
    help='예: {"평가 요소/ 평가 과제":"평가요소"}'
).strip()

try:
    extra_assess_aliases = json.loads(extra_assess_aliases_raw) if extra_assess_aliases_raw else {}
    if not isinstance(extra_assess_aliases, dict):
        st.sidebar.error("Extra ASSESS aliases must be a JSON object (dict).")
        extra_assess_aliases = {}
except Exception as e:
    st.sidebar.error(f"Invalid JSON for extra ASSESS aliases: {e}")
    extra_assess_aliases = {}

def group_by_subject(results: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for r in results:
        subj = r.get("_subject", "UNKNOWN")
        grouped.setdefault(subj, []).append(r)
    return grouped

# ---------- parse button (ONLY parse + store) ----------
if st.button("파싱 실행", type="primary"):
    if not uploaded:
        st.warning("파일을 먼저 업로드해줘.")
        st.stop()

    st.session_state.parsed_grouped_by_file = {}
    st.session_state.parsed_results_by_file = {}

    for f in uploaded:
        data = f.read()
        results = parse_2022(data, extra_assess_aliases=extra_assess_aliases) if mode == "2022" else []

        st.session_state.parsed_results_by_file[f.name] = results
        st.session_state.parsed_grouped_by_file[f.name] = group_by_subject(results)

    first_file = next(iter(st.session_state.parsed_grouped_by_file.keys()), None)
    st.session_state.selected_file = first_file
    if first_file:
        subjects = list(st.session_state.parsed_grouped_by_file[first_file].keys())
        st.session_state.selected_subject = subjects[0] if subjects else None

# ---------- render ----------
if not st.session_state.parsed_grouped_by_file:
    st.info("파일 업로드 후 '파싱 실행'을 눌러줘.")
    st.stop()

file_names = list(st.session_state.parsed_grouped_by_file.keys())

selected_file = st.selectbox(
    "결과를 볼 파일",
    file_names,
    index=file_names.index(st.session_state.selected_file) if st.session_state.selected_file in file_names else 0,
    key="selected_file"
)

grouped = st.session_state.parsed_grouped_by_file[selected_file]
results = st.session_state.parsed_results_by_file[selected_file]

st.markdown(f"## {selected_file}")
st.caption(f"subjects: {len(grouped)}개, total blocks: {len(results)}개")

st.sidebar.subheader("과목별 조회")

q = st.sidebar.text_input("과목 검색", value=st.session_state.subject_query, key="subject_query")
all_subjects = list(grouped.keys())
filtered_subjects = [s for s in all_subjects if q.strip() == "" or q.strip() in s]

show_all = st.sidebar.checkbox("전체 과목 한 번에 보기", value=st.session_state.show_all_subjects, key="show_all_subjects")

if not filtered_subjects:
    st.warning("검색 결과가 없어. 검색어를 지우거나 바꿔줘.")
    st.stop()

if st.session_state.selected_subject not in filtered_subjects:
    st.session_state.selected_subject = filtered_subjects[0]

selected_subject = st.sidebar.selectbox(
    "과목 선택",
    filtered_subjects,
    index=filtered_subjects.index(st.session_state.selected_subject),
    key="selected_subject"
)

subjects_to_render = filtered_subjects if show_all else [selected_subject]

for subj in subjects_to_render:
    items = grouped[subj]

    cleaned_items = []
    for it in items:
        it2 = dict(it)
        it2.pop("_subject", None)
        cleaned_items.append(it2)

    ach = [x for x in cleaned_items if "achievement_levels" in x]
    ass = [x for x in cleaned_items if "assessment_items" in x]

    with st.expander(
        f"과목: {subj} (achievement {len(ach)} / assessment {len(ass)} / total {len(cleaned_items)})",
        expanded=not show_all
    ):
        tab1, tab2, tab3 = st.tabs([
            f"Achievement ({len(ach)})",
            f"Assessment ({len(ass)})",
            f"All ({len(cleaned_items)})"
        ])
        with tab1:
            st.code(to_pretty_json(ach), language="json")
        with tab2:
            st.code(to_pretty_json(ass), language="json")
        with tab3:
            st.code(to_pretty_json(cleaned_items), language="json")
