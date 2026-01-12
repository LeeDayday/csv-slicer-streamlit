import streamlit as st

st.set_page_config(page_title="My Multi Tools", layout="centered")

page1 = st.Page("pages/csv-slicer_1.py", title="CSV Slicer", icon="📂", url_path="slicer-1", default=True)
page2 = st.Page("pages/slicer_2.py", title="Tool #2", icon="🧩", url_path="slicer-2")
page3 = st.Page("pages/slicer_3.py", title="Tool #3", icon="👩‍🏫", url_path="slicer-3")
nav = st.navigation([page1, page2, page3])
nav.run()
