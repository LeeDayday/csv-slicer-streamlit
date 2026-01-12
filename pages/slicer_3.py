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
    # HWPX에서 종종 깨져 나오는 기호들 대체
    s = s.replace("", "·")  # (서논술형) -> (서·논술형)
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# XML extraction helpers
# =========================
def extract_all_hp_text(elem: ET.Element, nsmap: Dict[str, str]) -> List[str]:
    hp_t = _ns("hp:t", nsmap)
    out = []
    for t in elem.iter(hp_t):
        if t.text is not None:
            txt = t.text.strip()
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
        parts = re.split(r"(?:^|\n)\s*[◦○•∙]\s*", "\n" + t)
        return [_clean_inline(p) for p in parts if p.strip()]
    # fallback: 줄바꿈
    return [_clean_inline(p) for p in t.split("\n") if p.strip()]


# =========================
# Title / Header configs
# =========================
ACH_TITLES = {"나. 단원/영역별 성취수준", "나. 영역별 성취수준"}
ASSESS_TITLES = {"다. 평가도구(예시)", "다. 예시 평가도구"}

SUBTITLE_RE = re.compile(r"^\(\d+\)\s*")  # ✅ subtitle 규칙: (숫자) 로 시작

ACH_HEADER_LEVEL = _normalize_header("성취수준")
ACH_HEADER_DESC = _normalize_header("일반적 특성")

# assessment canonical names
CAN_ITEM = _normalize_header("문항번호")
CAN_UNIT = _normalize_header("영역/단원명")
CAN_CODE = _normalize_header("교육과정성취기준코드")
CAN_TYPE = _normalize_header("문항유형")
CAN_ELEM = _normalize_header("평가요소")

ASSESS_HEADER_ALIASES = {
    _normalize_header("문항번호"): CAN_ITEM,
    _normalize_header("영역/단원명"): CAN_UNIT,

    _normalize_header("교육과정성취기준코드"): CAN_CODE,
    _normalize_header("교육과정 성취기준 코드"): CAN_CODE,
    _normalize_header("평가준거성취기준코드"): CAN_CODE,
    _normalize_header("평가준거 성취기준 코드"): CAN_CODE,
    _normalize_header("평가준거\n성취기준코드"): CAN_CODE,  # 줄바꿈이 헤더에 섞인 케이스도 normalize로 흡수됨

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
def parse_achievement_table(tbl: ET.Element, nsmap: Dict[str, str]) -> Optional[List[Dict[str, str]]]:
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix or len(matrix) < 2:
        return None

    header = [_normalize_header(x) for x in matrix[0]]
    if set(header) != {ACH_HEADER_LEVEL, ACH_HEADER_DESC}:
        return None

    col_level = header.index(ACH_HEADER_LEVEL)
    col_desc = header.index(ACH_HEADER_DESC)

    items = []
    for r in matrix[1:]:
        if len(r) <= max(col_level, col_desc):
            continue
        level = _clean_inline(r[col_level])
        desc = _normalize_text(r[col_desc])  # 성취수준 설명은 줄바꿈 유지가 유리할 수도 있어 유지
        desc = re.sub(r"\s+", " ", desc.replace("\n", " ")).strip()  # 결국 출력은 한 줄로
        if not level and not desc:
            continue
        items.append({"level": level, "description": desc})

    return items if len(items) >= 3 else None

def _canonize_header(h: str) -> str:
    nh = _normalize_header(h)
    return ASSESS_HEADER_ALIASES.get(nh, nh)

def parse_assessment_table(tbl: ET.Element, nsmap: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
    matrix = table_to_matrix(tbl, nsmap)
    if not matrix or len(matrix) < 2:
        return None

    header_canon = [_canonize_header(h) for h in matrix[0]]
    if not REQUIRED_ASSESS_CANON.issubset(set(header_canon)):
        return None

    def idx(can_name: str) -> int:
        return header_canon.index(can_name)

    col_item = idx(CAN_ITEM)
    col_unit = idx(CAN_UNIT)
    col_code = idx(CAN_CODE)
    col_type = idx(CAN_TYPE)
    col_elem = idx(CAN_ELEM)

    items = []
    for r in matrix[1:]:
        if len(r) <= max(col_item, col_unit, col_code, col_type, col_elem):
            continue

        item_number = _clean_inline(r[col_item])
        domain_unit = _clean_inline(r[col_unit])
        curriculum_code = _clean_inline(r[col_code])
        assessment_type = _clean_inline(r[col_type])  # ✅ 여기서 \n 제거 +  → ·
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
# ✅ 핵심: "바깥 hp:p"만 순회 (표 안쪽 hp:p 제외)
# =========================
def iter_outer_paragraphs(root: ET.Element, nsmap: Dict[str, str]):
    hp_p = _ns("hp:p", nsmap)
    hp_tbl = _ns("hp:tbl", nsmap)
    hp_tc = _ns("hp:tc", nsmap)

    # parent map 구성
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
        # ✅ 표 내부(tc/tbl)의 hp:p는 제외
        if has_ancestor(p, {hp_tbl, hp_tc}):
            continue
        yield p


# =========================
# ✅ Main parser: title 1개 아래 (subtitle→table)* 전부 누적
# =========================
def parse_sections_from_section_xml(section_xml: str) -> List[Dict[str, Any]]:
    """
    반환(평탄화):
    [
      {"title": "나. ...", "subtitle": "(1)...", "achievement_levels":[...]},
      {"title": "나. ...", "subtitle": "(2)...", "achievement_levels":[...]},
      ...
      {"title": "다. ...", "subtitle": "(1)...", "assessment_items":[...]},
      ...
    ]
    """
    root = ET.fromstring(section_xml)
    nsmap = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
    hp_tbl = _ns("hp:tbl", nsmap)

    results: List[Dict[str, Any]] = []

    current_title: Optional[str] = None
    current_type: Optional[str] = None  # "ach" | "assess"
    pending_subtitle: Optional[str] = None

    for p in iter_outer_paragraphs(root, nsmap):
        tbl = p.find(".//" + hp_tbl)

        # (A) 표가 있는 문단이면: subtitle + table을 한 객체로 append
        if tbl is not None:
            if not (current_title and current_type and pending_subtitle):
                continue

            if current_type == "ach":
                items = parse_achievement_table(tbl, nsmap)
                if items:
                    results.append({
                        "title": current_title,
                        "subtitle": pending_subtitle,
                        "achievement_levels": items
                    })
                    pending_subtitle = None

            elif current_type == "assess":
                items = parse_assessment_table(tbl, nsmap)
                if items:
                    results.append({
                        "title": current_title,
                        "subtitle": pending_subtitle,
                        "assessment_items": items
                    })
                    pending_subtitle = None

            continue

        # (B) 일반 문단이면: title/subtitle 감지
        texts = extract_all_hp_text(p, nsmap)
        if not texts:
            continue

        for txt in texts:
            txt = txt.strip()

            # title start (여기서 ACH_TITLES처럼 set을 쓰면 in으로)
            if txt in ACH_TITLES:  # 예: {"나. 단원/영역별 성취수준","나. 영역별 성취수준"}
                current_title = "나. 단원/영역별 성취수준"  # 대표 title로 통일(원하면 txt로)
                current_type = "ach"
                pending_subtitle = None
                continue

            if txt in ASSESS_TITLES:  # {"다. 평가도구(예시)","다. 예시 평가도구"}
                current_title = txt  # 혹은 대표값으로 통일 가능
                current_type = "assess"
                pending_subtitle = None
                continue

            # subtitle
            if current_title and SUBTITLE_RE.match(txt):
                pending_subtitle = txt
                continue

    return results



# =========================
# HWPX loader
# =========================
def read_hwpx_sections_from_bytes(hwpx_bytes: bytes) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(hwpx_bytes), "r") as z:
        section_names = sorted([n for n in z.namelist() if re.match(r"^Contents/section\d+\.xml$", n)])
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
    "파일 업로드 (.hwpx 또는 Contents_section*.xml(.txt))",
    type=["hwpx", "txt", "xml"],
    accept_multiple_files=True
)

if st.button("파싱 실행", type="primary"):
    if not uploaded:
        st.warning("파일을 먼저 업로드해줘.")
        st.stop()

    results: List[Dict[str, Any]] = []
    for f in uploaded:
        name = f.name
        data = f.read()

        if name.lower().endswith(".hwpx"):
            results.extend(parse_hwpx_bytes(data))
        else:
            xml_text = data.decode("utf-8", errors="replace")
            results.extend(parse_sections_from_section_xml(xml_text))

    json_text = json.dumps(results, ensure_ascii=False, indent=2)

    st.subheader("JSON 미리보기")
    st.code(json_text, language="json")
