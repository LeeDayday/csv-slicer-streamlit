# app.py
import re
import json
import zipfile
import io
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

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
    """줄바꿈 제거 + 깨진 글자 대체 등 출력용 정리"""
    s = _normalize_text(s).replace("\n", " ").strip()
    s = s.replace("", "·")  # (서논술형) -> (서·논술형)
    s = re.sub(r"\s+", " ", s)
    return s

# --- add: heading helpers (near configs) ---
HEADING_MAX_LEN = 80  # 너무 긴 문단은 제목/소제목 후보에서 제외(오탐 방지)

RESET_SECTION_PREFIXES = (
    "가.", "가. 교육과정", "가. 교육과정 성취기준",
    "참 고", "참고", "부록", "참 고 문 헌", "참고 문헌"
)

def _norm_for_match(s: str) -> str:
    # 공백/개행 제거 + 흔한 점(.) 변형 통일(필요시 추가)
    s = _clean_inline(s)
    s = s.replace("．", ".")
    return _normalize_header(s)

def detect_title_type(para_text: str) -> Optional[Tuple[str, str]]:
    """
    return (type, original_title_text)
    type: "ach" | "assess"
    """
    t_norm = _norm_for_match(para_text)

    # 1) exact set match (공백/개행 차이 흡수)
    for t in ACH_TITLES:

        if _norm_for_match(t) == t_norm:
            return ("ach", _clean_inline(para_text))
    for t in ASSESS_TITLES:
        if _norm_for_match(t) == t_norm:
            return ("assess", _clean_inline(para_text))

    # 2) heuristic fallback (세트에 없는 변형 제목도 잡기)
    clean = _clean_inline(para_text)
    if clean.startswith("나.") and ("성취" in clean and "수준" in clean):
        return ("ach", clean)
    if clean.startswith("다.") and ("평가" in clean and "도구" in clean):
        return ("assess", clean)

    return None

# =========================
# XML extraction helpers
# =========================
def extract_all_hp_text(elem: ET.Element, nsmap: Dict[str, str]) -> List[str]:
    """
    ✅ 핵심 수정:
    hp:t 내부에 <hp:fwSpace/> 같은 자식이 끼면,
    t.text에는 '◦'만 있고 실제 문장은 child.tail로 들어간다.
    따라서 itertext()로 text+tail을 모두 수집해야 한다.
    """
    hp_t = _ns("hp:t", nsmap)
    out: List[str] = []
    for t in elem.iter(hp_t):
        txt = "".join(t.itertext())  # ✅ text + tail 모두 포함
        txt = txt.strip()
        if txt:
            out.append(txt)
    return out

def extract_cell_text(tc: ET.Element, nsmap: Dict[str, str]) -> str:
    parts = extract_all_hp_text(tc, nsmap)
    return _normalize_text("\n".join(parts))

def table_to_matrix(tbl: ET.Element, nsmap: Dict[str, str]) -> List[List[str]]:
    hp_tr = _ns("hp:tr", nsmap)
    hp_tc = _ns("hp:tc", nsmap)

    rows = []
    for tr in tbl.iter(hp_tr):
        row = []
        for tc in tr.findall(hp_tc):
            row.append(extract_cell_text(tc, nsmap))
        if row:
            rows.append(row)
    return rows

def split_eval_elements(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    # bullet 분리
    if any(b in t for b in ["◦", "○", "•", "∙"]):
        # bullet이 줄 중간에 붙어도 분리되도록, 각 줄 시작 기준 분리
        parts = re.split(r"(?:^|\n)\s*[◦○•∙]\s*", "\n" + t)
        return [_clean_inline(p) for p in parts if p.strip()]
    # fallback: 줄바꿈
    return [_clean_inline(p) for p in t.split("\n") if p.strip()]



## =========================
# Title / Header configs
# =========================
ACH_TITLES = {
    "나. 단원/영역별 성취수준", 
    "나. 영역별 성취수준", 
    "나. 단원별 성취수준 예시",
    "나. 학기 단위 (과목 단위) 성취수준",
    "2. 단원별 성취수준"

}
ASSESS_TITLES = {"다. 평가도구(예시)", "다. 예시 평가도구", "다. 평가도구(예시)"}

SUBTITLE_RE = re.compile(r"^\(\d+\)\s*")  # subtitle 규칙: (숫자) 로 시작

# ✅ achievement table header aliases
ACH_LEVEL_HEADERS = {
    _normalize_header("성취수준"),
    _normalize_header("수준"),
}

ACH_DESC_HEADERS = {
    _normalize_header("일반적 특성"),
    _normalize_header("일반적 특징")
    # 필요하면 여기도 변형 케이스 추가
    # _normalize_header("일반적특성"),
}

# assessment canonical names
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
    _normalize_header("평가준거\n성취기준코드"): CAN_CODE,

    _normalize_header("문항유형"): CAN_TYPE,
    _normalize_header("문항 유형"): CAN_TYPE,
    _normalize_header("문장유형"): CAN_TYPE,
    _normalize_header("문장 유형"): CAN_TYPE,

    _normalize_header("평가요소"): CAN_ELEM,
    _normalize_header("평가 요소"): CAN_ELEM,
}

REQUIRED_ASSESS_CANON = {CAN_ITEM, CAN_UNIT, CAN_CODE, CAN_TYPE, CAN_ELEM}


# =========================
# Table parsers
# =========================
def _find_header_row_for_achievement(matrix: List[List[str]], scan_rows: int = 6) -> Optional[Tuple[int, int, int]]:
    """
    ✅ 캡션/설명 행이 표 맨 위에 들어가는 케이스 대응:
    앞에서 몇 줄 스캔하면서 헤더(수준/일반적특성)를 찾는다.
    """
    for r_idx in range(min(scan_rows, len(matrix))):
        header = [_normalize_header(x) for x in matrix[r_idx]]
        col_level = next((i for i, h in enumerate(header) if h in ACH_LEVEL_HEADERS), None)
        col_desc  = next((i for i, h in enumerate(header) if h in ACH_DESC_HEADERS), None)
        if col_level is not None and col_desc is not None:
            return (r_idx, col_level, col_desc)
    return None
def parse_achievement_table(tbl, nsmap):
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix:
        return None

    for r_idx in range(min(6, len(matrix))):
        header = [_normalize_header(x) for x in matrix[r_idx]]
        col_level = next((i for i,h in enumerate(header) if h in ACH_LEVEL_HEADERS), None)
        col_desc  = next((i for i,h in enumerate(header) if h in ACH_DESC_HEADERS), None)

        if col_level is not None and col_desc is not None:
            items = []
            for row in matrix[r_idx+1:]:
                if len(row) <= max(col_level, col_desc):
                    continue
                level = _clean_inline(row[col_level])
                desc  = _clean_inline(row[col_desc])
                if level or desc:
                    items.append({"level": level, "description": desc})
            return items if items else None

    return None


def _canonize_header(h: str) -> str:
    nh = _normalize_header(h)
    return ASSESS_HEADER_ALIASES.get(nh, nh)

def _find_header_row_for_assessment(matrix: List[List[str]], scan_rows: int = 6) -> Optional[int]:
    """
    (선택) 평가도구 표도 캡션행이 끼는 케이스가 있으니
    REQUIRED_ASSESS_CANON이 만족되는 헤더 행을 스캔해서 찾는다.
    """
    for r_idx in range(min(scan_rows, len(matrix))):
        header_canon = [_canonize_header(h) for h in matrix[r_idx]]
        if REQUIRED_ASSESS_CANON.issubset(set(header_canon)):
            return r_idx
    return None


def parse_assessment_table(tbl, nsmap):
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix:
        return None

    for r_idx in range(min(6, len(matrix))):
        header = [_canonize_header(h) for h in matrix[r_idx]]
        if REQUIRED_ASSESS_CANON.issubset(set(header)):
            items = []
            for row in matrix[r_idx+1:]:
                if len(row) < len(header):
                    continue
                items.append({
                    "item_number": _clean_inline(row[header.index(CAN_ITEM)]),
                    "domain_unit": _clean_inline(row[header.index(CAN_UNIT)]),
                    "curriculum_code": _clean_inline(row[header.index(CAN_CODE)]),
                    "assessment_type": _clean_inline(row[header.index(CAN_TYPE)]),
                    "evaluation_elements": split_eval_elements(row[header.index(CAN_ELEM)])
                })
            return items if items else None
    return None



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
# Main parser
# =========================
def parse_sections_from_section_xml(section_xml: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(section_xml)
    nsmap = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
    hp_tbl = _ns("hp:tbl", nsmap)

    results: List[Dict[str, Any]] = []

    # ✅ “있으면” 쓰고, 없으면 ""로 처리할 컨텍스트
    last_title_type: Optional[str] = None   # "ach" | "assess"
    last_title_text: str = ""              # 규칙 맞으면 채움
    pending_subtitle: str = ""             # 규칙 맞으면 채움

    for p in iter_outer_paragraphs(root, nsmap):
        tbl = p.find(".//" + hp_tbl)

        # =========================
        # (A) 표 문단: ✅ 무조건 헤더로 판별/파싱
        # =========================
        if tbl is not None:
            ach_items = parse_achievement_table(tbl, nsmap)
            if ach_items is not None:
                title = last_title_text if last_title_type == "ach" else ""
                subtitle = pending_subtitle  # 없으면 이미 ""
                results.append({
                    "title": title,
                    "subtitle": subtitle,
                    "achievement_levels": ach_items
                })
                pending_subtitle = ""  # 표 하나에 subtitle 소비
                continue

            ass_items = parse_assessment_table(tbl, nsmap)
            if ass_items is not None:
                title = last_title_text if last_title_type == "assess" else ""
                subtitle = pending_subtitle
                results.append({
                    "title": title,
                    "subtitle": subtitle,
                    "assessment_items": ass_items
                })
                pending_subtitle = ""
                continue

            # ach/ass 헤더가 아니면 무시
            continue

        # =========================
        # (B) 일반 문단: title/subtitle만 “규칙 맞으면” 저장
        # =========================
        texts = extract_all_hp_text(p, nsmap)
        if not texts:
            continue

        para_text = _clean_inline(" ".join(texts))
        if not para_text:
            continue

        # 큰 섹션 리셋(원래 로직 유지)
        if para_text.startswith(RESET_SECTION_PREFIXES):
            last_title_type = None
            last_title_text = ""
            pending_subtitle = ""
            continue

        # 너무 긴 문단은 heading 후보에서 제외(원래 기준 유지)
        if len(para_text) <= HEADING_MAX_LEN:
            # title: detect 규칙에 맞을 때만 반영, 아니면 그대로(= 공백 유지 가능)
            hit = detect_title_type(para_text)
            if hit:
                last_title_type, last_title_text = hit  # 규칙 맞는 제목만 저장
                pending_subtitle = ""                   # 새 title 시작이면 subtitle 초기화
                continue

            # subtitle: (숫자) 규칙에 맞을 때만 저장
            if SUBTITLE_RE.match(para_text):
                pending_subtitle = para_text
                continue

        # 그 외 텍스트는 title/subtitle로 취급하지 않음(= "" 유지)
        continue

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

def parse_hwpx_bytes(hwpx_bytes: bytes) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for _, xml_text in read_hwpx_sections_from_bytes(hwpx_bytes):
        all_results.extend(parse_sections_from_section_xml(xml_text))
    return all_results


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="HWPX 파서", layout="wide")
st.title("HWPX > JSON (단원영역별 성취수준, 평가예시)")

uploaded = st.file_uploader(
    "파일 업로드 (.hwpx)",
    type=["hwpx"],
    accept_multiple_files=True
)

if st.button("파싱 실행", type="primary"):
    if not uploaded:
        st.warning("파일을 먼저 업로드해줘.")
        st.stop()

    all_files_results: Dict[str, List[Dict[str, Any]]] = {}

    for f in uploaded:
        name = f.name
        data = f.read()
        all_files_results[name] = parse_hwpx_bytes(data)

    # 화면 표시(다운로드 없이)
    st.subheader("파일별 결과")
    for fname, results in all_files_results.items():
        ach_cnt = sum(1 for x in results if "achievement_levels" in x)
        ass_cnt = sum(1 for x in results if "assessment_items" in x)
        st.markdown(f"### {fname}")
        st.caption(f"achievement: {ach_cnt}개, assessment: {ass_cnt}개, total: {len(results)}개")
        tab1, tab2, tab3 = st.tabs(["Achievement", "Assessment", "All"])

        with tab1:
            st.json([r for r in results if "achievement_levels" in r])

        with tab2:
            st.json([r for r in results if "assessment_items" in r])

        with tab3:
            st.json(results)

