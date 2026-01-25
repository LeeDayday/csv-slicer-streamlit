# app.py
import re
import json
import zipfile
import io
import textwrap
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

import streamlit as st


# =========================
# Utils
# =========================
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
    s = s.replace("", "·")      # 깨진 글머리/기호 보정
    s = s.replace("ㆍ", "･")      # 서ㆍ논술형 -> 서･논술형 (원하면 제거)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_table_caption(s: str) -> bool:
    s = _clean_inline(s)
    return s.startswith("<표")

def split_eval_elements(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    # bullet 분리
    if any(b in t for b in ["◦", "○", "•", "∙"]):
        parts = re.split(r"(?:^|\n)\s*[◦○•∙]\s*", "\n" + t)
        return [_clean_inline(p) for p in parts if p.strip()]
    # 2022는 bullet 없이 띄어쓰기만 있는 경우가 많아서 줄바꿈 기준 fallback
    return [_clean_inline(p) for p in t.split("\n") if p.strip()]

def to_pretty_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def wrap_long_lines(s: str, width: int = 120) -> str:
    out_lines = []
    for line in s.splitlines():
        if len(line) <= width:
            out_lines.append(line)
        else:
            out_lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(out_lines)


# =========================
# Title / Header configs
# =========================
SUBTITLE_RE = re.compile(r"^\(\d+\)\s*")  # (1) ...

# (2015용) achievement table header aliases
ACH_LEVEL_HEADERS = {_normalize_header("성취수준"), _normalize_header("수준")}
ACH_DESC_HEADERS  = {_normalize_header("일반적 특성"), _normalize_header("일반적 특징")}

# ---- assessment canonical names (내부 표준키) ----
CAN_ITEM = _normalize_header("문항번호")
CAN_UNIT = _normalize_header("영역/단원명")
CAN_CODE = _normalize_header("교육과정성취기준코드")
CAN_TYPE = _normalize_header("문항유형")
CAN_ELEM = _normalize_header("평가요소")

# ---- 2015 + 2022 헤더 alias 통합 ----
ASSESS_HEADER_ALIASES = {
    # 2015
    _normalize_header("문항번호"): CAN_ITEM,
    _normalize_header("문항 번호"): CAN_ITEM,

    _normalize_header("영역/단원명"): CAN_UNIT,
    _normalize_header("영역"): CAN_UNIT,

    _normalize_header("교육과정성취기준코드"): CAN_CODE,
    _normalize_header("교육과정 성취기준 코드"): CAN_CODE,
    _normalize_header("평가준거성취기준코드"): CAN_CODE,
    _normalize_header("평가준거 성취기준 코드"): CAN_CODE,
    _normalize_header("평가준거\n성취기준코드"): CAN_CODE,

    _normalize_header("문항유형"): CAN_TYPE,
    _normalize_header("문항 유형"): CAN_TYPE,

    _normalize_header("평가요소"): CAN_ELEM,
    _normalize_header("평가 요소"): CAN_ELEM,

    # 2022 (샘플 기준)
    _normalize_header("번호"): CAN_ITEM,
    _normalize_header("성취기준"): CAN_CODE,
    _normalize_header("평가도구유형"): CAN_TYPE,
    _normalize_header("평가 도구 유형"): CAN_TYPE,
    _normalize_header("평가요소/평가과제"): CAN_ELEM,
    _normalize_header("평가 요소/ 평가 과제"): CAN_ELEM,
    _normalize_header("평가요소/ 평가과제"): CAN_ELEM,
    _normalize_header("평가 요소/평가 과제"): CAN_ELEM,
}

REQUIRED_ASSESS_CANON = {CAN_ITEM, CAN_UNIT, CAN_CODE, CAN_TYPE, CAN_ELEM}


# =========================
# Heading helpers
# =========================
HEADING_MAX_LEN = 120

RESET_SECTION_PREFIXES = (
    "참 고", "참고", "부록", "참 고 문 헌", "참고 문헌"
)

# 2022에서 타이틀을 “통일”하려는 요구 반영
ACH_TITLE_2022_CANON = "2 영역별 성취수준"
ASS_TITLE_2022_CANON = "3 예시 평가 도구"

# 정규화용 패턴 (괄호 소제목 붙어도 타이틀만 뽑기)
RE_ACH_TITLE_2022 = re.compile(r"^\s*2\s*영역별\s*성취수준\b")
RE_ASS_TITLE_2022 = re.compile(r"^\s*3\s*예시\s*평가\s*도구\b")

def pick_title_from_context(ctx: List[str], table_type: str) -> str:
    """
    1) 2022 타이틀을 발견하면 "2 영역별 성취수준" / "3 예시 평가 도구"로 통일
    2) 못 찾으면 힌트 기반 fallback
    """
    for t in reversed(ctx):
        clean = _clean_inline(t)
        if _is_table_caption(clean):
            continue

        if table_type == "ach" and RE_ACH_TITLE_2022.search(clean):
            return ACH_TITLE_2022_CANON
        if table_type == "assess" and RE_ASS_TITLE_2022.search(clean):
            return ASS_TITLE_2022_CANON

    # fallback: 힌트 기반
    hints = ("성취수준", "성취 수준") if table_type == "ach" else ("평가도구", "평가 도구", "예시 평가 도구")
    for t in reversed(ctx):
        clean = _clean_inline(t)
        if _is_table_caption(clean):
            continue
        if any(h in clean for h in hints):
            # 2022면 굳이 긴 문장 저장하지 말고 통일
            if table_type == "ach" and "영역별" in clean and "성취" in clean:
                return ACH_TITLE_2022_CANON
            if table_type == "assess" and ("예시" in clean and "평가" in clean):
                return ASS_TITLE_2022_CANON
            return clean

    return ""

def pick_subtitle_from_context(ctx: List[str]) -> str:
    for t in reversed(ctx):
        clean = _clean_inline(t)
        if _is_table_caption(clean):
            continue
        if SUBTITLE_RE.match(clean):
            return clean
    return ""


# =========================
# XML extraction helpers
# =========================
def extract_all_hp_text(elem: ET.Element, nsmap: Dict[str, str]) -> List[str]:
    hp_t = _ns("hp:t", nsmap)
    out: List[str] = []
    for t in elem.iter(hp_t):
        txt = "".join(t.itertext())  # text + tail 포함
        txt = txt.strip()
        if txt:
            out.append(txt)
    return out

def extract_cell_text(tc: ET.Element, nsmap: Dict[str, str]) -> str:
    parts = extract_all_hp_text(tc, nsmap)
    return _normalize_text(" ".join(parts))

def table_to_matrix(tbl: ET.Element, nsmap: Dict[str, str]) -> List[List[str]]:
    hp_tr = _ns("hp:tr", nsmap)
    hp_tc = _ns("hp:tc", nsmap)

    rows: List[List[str]] = []
    for tr in tbl.iter(hp_tr):
        row = []
        for tc in tr.findall(hp_tc):
            row.append(extract_cell_text(tc, nsmap))
        if row:
            rows.append(row)
    return rows


# =========================
# "바깥 hp:p"만 순회 (표 안쪽 hp:p 제외)
# =========================
def iter_outer_paragraphs(root: ET.Element, nsmap: Dict[str, str]):
    hp_p = _ns("hp:p", nsmap)
    hp_tbl = _ns("hp:tbl", nsmap)
    hp_tc = _ns("hp:tc", nsmap)

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

    for p in root.iter(hp_p):
        if has_ancestor(p, {hp_tbl, hp_tc}):
            continue
        yield p


# =========================
# 2022 Achievement parser (category + level + description)
# =========================
LEVEL_RE = re.compile(r"^[A-E]$")

def _is_level(s: str) -> bool:
    s = _clean_inline(s)
    return bool(LEVEL_RE.match(s))

def _norm_category(s: str) -> Optional[str]:
    s = _clean_inline(s).replace("·", "･")
    if "지식" in s:
        return "지식･이해"
    if "과정" in s:
        return "과정･기능"
    if "가치" in s:
        return "가치･태도"
    return None

def parse_achievement_table_2022(tbl: ET.Element, nsmap: Dict[str, str]) -> Optional[List[Dict[str, str]]]:
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix or len(matrix) < 2:
        return None

    # 2022 성취수준 표 판별(가볍게)
    flat = " ".join(_clean_inline(x) for row in matrix[:3] for x in row if x)
    if ("영역별" not in flat) or ("성취수준" not in flat and "성취 수준" not in flat):
        return None
    if not any(_is_level(x) for row in matrix[:12] for x in row if x):
        return None

    items: List[Dict[str, str]] = []
    current_level: Optional[str] = None

    for row in matrix:
        row_cells = [_clean_inline(x) for x in row]
        row_cells = [x for x in row_cells if x]
        if not row_cells:
            continue

        # 레벨 행(예: [..., 'A', '지식･이해', '설명'])
        level_idx = None
        for i, cell in enumerate(row_cells[:5]):
            if _is_level(cell):
                level_idx = i
                break

        if level_idx is not None:
            current_level = row_cells[level_idx]
            cat = None
            desc = ""

            if level_idx + 1 < len(row_cells):
                cat = _norm_category(row_cells[level_idx + 1])
            if cat and level_idx + 2 < len(row_cells):
                desc = _clean_inline(" ".join(row_cells[level_idx + 2:]))
            else:
                desc = _clean_inline(" ".join(row_cells[level_idx + 1:]))

            if cat and desc:
                items.append({"category": cat, "level": current_level, "description": desc})
            continue

        # category-only 행(예: ['과정･기능', '설명...'])
        if len(row_cells) >= 2:
            cat = _norm_category(row_cells[0])
            if cat and current_level:
                desc = _clean_inline(" ".join(row_cells[1:]))
                if desc:
                    items.append({"category": cat, "level": current_level, "description": desc})
                continue

    return items if len(items) >= 3 else None


# =========================
# 2015 Achievement parser (header-based)
# =========================
def parse_achievement_table_2015(
    tbl: ET.Element,
    nsmap: Dict[str, str],
    extra_level_headers: Optional[List[str]] = None,
    extra_desc_headers: Optional[List[str]] = None,
) -> Optional[List[Dict[str, str]]]:
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix or len(matrix) < 2:
        return None

    level_headers = set(ACH_LEVEL_HEADERS)
    desc_headers = set(ACH_DESC_HEADERS)

    if extra_level_headers:
        level_headers |= {_normalize_header(x) for x in extra_level_headers if x}
    if extra_desc_headers:
        desc_headers |= {_normalize_header(x) for x in extra_desc_headers if x}

    for r_idx in range(min(6, len(matrix))):
        header = [_normalize_header(x) for x in matrix[r_idx]]
        col_level = next((i for i, h in enumerate(header) if h in level_headers), None)
        col_desc = next((i for i, h in enumerate(header) if h in desc_headers), None)

        if col_level is not None and col_desc is not None:
            items = []
            for row in matrix[r_idx + 1:]:
                if len(row) <= max(col_level, col_desc):
                    continue
                level = _clean_inline(row[col_level])
                desc = _clean_inline(row[col_desc])
                if level or desc:
                    items.append({"level": level, "description": desc})
            return items if items else None

    return None

def parse_achievement_table(
    tbl: ET.Element,
    nsmap: Dict[str, str],
    extra_level_headers: Optional[List[str]] = None,
    extra_desc_headers: Optional[List[str]] = None,
) -> Optional[List[Dict[str, str]]]:
    # 2022 우선
    items_2022 = parse_achievement_table_2022(tbl, nsmap)
    if items_2022 is not None:
        return items_2022
    # fallback 2015
    return parse_achievement_table_2015(
        tbl, nsmap,
        extra_level_headers=extra_level_headers,
        extra_desc_headers=extra_desc_headers
    )


# =========================
# Assessment parser (2015/2022 공용)
# =========================
def parse_assessment_table(
    tbl: ET.Element,
    nsmap: Dict[str, str],
    extra_aliases: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix or len(matrix) < 2:
        return None

    aliases = dict(ASSESS_HEADER_ALIASES)
    if extra_aliases:
        for k, v in extra_aliases.items():
            if not k or not v:
                continue
            aliases[_normalize_header(k)] = _normalize_header(v)

    def canon(h: str) -> str:
        nh = _normalize_header(h)
        return aliases.get(nh, nh)

    # 헤더 행 찾기 (앞쪽 몇 줄 스캔)
    header_row_idx: Optional[int] = None
    scan_rows = min(8, len(matrix))
    for r_idx in range(scan_rows):
        header_canon = [canon(h) for h in matrix[r_idx]]
        if REQUIRED_ASSESS_CANON.issubset(set(header_canon)):
            header_row_idx = r_idx
            break

    if header_row_idx is None:
        return None

    header_canon = [canon(h) for h in matrix[header_row_idx]]

    def idx(can_name: str) -> int:
        return header_canon.index(can_name)

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
# Main parser per section.xml
# =========================
def parse_sections_from_section_xml(
    section_xml: str,
    section_no: int = 0,
    extra_ach_desc_headers: Optional[List[str]] = None,
    extra_assess_aliases: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    root = ET.fromstring(section_xml)

    # HWPX 2011 paragraph ns
    nsmap = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
    hp_tbl = _ns("hp:tbl", nsmap)

    results: List[Dict[str, Any]] = []
    recent_paras = deque(maxlen=20)

    para_no = -1
    for p in iter_outer_paragraphs(root, nsmap):
        para_no += 1
        tbl = p.find(".//" + hp_tbl)

        # (A) 표 문단
        if tbl is not None:
            ctx = list(recent_paras)

            ach_items = parse_achievement_table(
                tbl, nsmap,
                extra_desc_headers=extra_ach_desc_headers
            )
            if ach_items is not None:
                title = pick_title_from_context(ctx, "ach")
                subtitle = pick_subtitle_from_context(ctx)
                results.append({
                    "_order": (section_no, para_no),
                    "title": title,
                    "subtitle": subtitle,
                    "achievement_levels": ach_items
                })
                continue

            ass_items = parse_assessment_table(
                tbl, nsmap,
                extra_aliases=extra_assess_aliases
            )
            if ass_items is not None:
                title = pick_title_from_context(ctx, "assess")
                subtitle = pick_subtitle_from_context(ctx)
                results.append({
                    "_order": (section_no, para_no),
                    "title": title,
                    "subtitle": subtitle,
                    "assessment_items": ass_items
                })
                continue

            continue  # ach/ass 둘 다 아니면 무시

        # (B) 일반 문단: context에 누적
        texts = extract_all_hp_text(p, nsmap)
        if not texts:
            continue

        para_text = _clean_inline(" ".join(texts))
        if not para_text:
            continue

        if para_text.startswith(RESET_SECTION_PREFIXES):
            recent_paras.clear()
            continue

        if len(para_text) <= HEADING_MAX_LEN:
            recent_paras.append(para_text)

    return results


# =========================
# HWPX loader (section 순서 보장)
# =========================
_SECTION_RE = re.compile(r"^Contents/section(\d+)\.xml$")

def _section_sort_key(name: str) -> Tuple[int, str]:
    m = _SECTION_RE.match(name)
    if not m:
        return (10**9, name)
    return (int(m.group(1)), name)

def read_hwpx_sections_from_bytes(hwpx_bytes: bytes) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(hwpx_bytes), "r") as z:
        section_names = [n for n in z.namelist() if _SECTION_RE.match(n)]
        section_names.sort(key=_section_sort_key)

        out = []
        for name in section_names:
            xml_text = z.read(name).decode("utf-8", errors="replace")
            out.append((name, xml_text))
        return out

def parse_hwpx_bytes(
    hwpx_bytes: bytes,
    extra_ach_desc_headers: Optional[List[str]] = None,
    extra_assess_aliases: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    sections = read_hwpx_sections_from_bytes(hwpx_bytes)

    for sec_idx, (_, xml_text) in enumerate(sections):
        all_results.extend(
            parse_sections_from_section_xml(
                xml_text,
                section_no=sec_idx,
                extra_ach_desc_headers=extra_ach_desc_headers,
                extra_assess_aliases=extra_assess_aliases,
            )
        )

    all_results.sort(key=lambda x: x.get("_order", (10**9, 10**9)))
    for r in all_results:
        r.pop("_order", None)

    return all_results


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="HWPX 파서 (2015/2022)", layout="wide")

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

st.title("HWPX > JSON (2 영역별 성취수준, 3 예시 평가 도구)")

uploaded = st.file_uploader(
    "파일 업로드 (.hwpx)",
    type=["hwpx"],
    accept_multiple_files=True
)

st.sidebar.header("Header overrides (optional)")

extra_ach_desc_raw = st.sidebar.text_input(
    "Extra ACH desc headers (comma-separated)",
    value="",
).strip()
extra_ach_desc_headers = [x.strip() for x in extra_ach_desc_raw.split(",") if x.strip()]

extra_assess_aliases_raw = st.sidebar.text_area(
    "Extra ASSESS header aliases (JSON dict)",
    value="{}",
    height=140,
    help='예: {"평가 요소/ 평가 과제":"평가요소"} (이미 2022 기본 alias 포함됨)'
).strip()

try:
    extra_assess_aliases = json.loads(extra_assess_aliases_raw) if extra_assess_aliases_raw else {}
    if not isinstance(extra_assess_aliases, dict):
        st.sidebar.error("Extra ASSESS aliases must be a JSON object (dict).")
        extra_assess_aliases = {}
except Exception as e:
    st.sidebar.error(f"Invalid JSON for extra ASSESS aliases: {e}")
    extra_assess_aliases = {}


if st.button("파싱 실행", type="primary"):
    if not uploaded:
        st.warning("파일을 먼저 업로드해줘.")
        st.stop()

    all_files_results: Dict[str, List[Dict[str, Any]]] = {}

    for f in uploaded:
        data = f.read()
        all_files_results[f.name] = parse_hwpx_bytes(
            data,
            extra_ach_desc_headers=extra_ach_desc_headers,
            extra_assess_aliases=extra_assess_aliases,
        )

    st.subheader("파일별 결과 (st.code 출력)")

    for fname, results in all_files_results.items():
        ach = [r for r in results if "achievement_levels" in r]
        ass = [r for r in results if "assessment_items" in r]

        st.markdown(f"### {fname}")
        st.caption(f"achievement: {len(ach)}개, assessment: {len(ass)}개, total: {len(results)}개")

        tab1, tab2, tab3 = st.tabs(["Achievement", "Assessment", "All"])

        with tab1:
            st.code(to_pretty_json(ach), language="json")

        with tab2:
            st.code(to_pretty_json(ass), language="json")

        with tab3:
            st.code(to_pretty_json(results), language="json")
else:
    st.info("좌측에서 .hwpx 파일을 업로드한 뒤, '파싱 실행'을 눌러줘.")
