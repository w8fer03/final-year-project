"""Section anchors for the single dashboard document.

The small local component scrolls the existing document only. It does not fetch
resources, navigate routes, or conditionally hide any dashboard content.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

SECTION_ANCHORS = {
    "Home": "sentilokal-home",
    "Dashboard Overview": "sentilokal-overview",
    "Product Sentiment Analysis": "sentilokal-products",
    "Customer Review Explorer": "sentilokal-explorer",
    "Model Performance": "sentilokal-performance",
}


def section_anchor(section: str):
    anchor = SECTION_ANCHORS[section]
    st.markdown(f'<div id="{anchor}" class="section-anchor" tabindex="-1"></div>', unsafe_allow_html=True)


def scroll_to_section(section: str, revision: int):
    # Only application-owned destinations enter the script, never uploaded text.
    anchor = SECTION_ANCHORS[section]
    components.html(
        """<script>
        const anchor = """ + json.dumps(anchor) + ";\nconst revision = " + str(int(revision)) + ";" + """
        let attempts = 0;
        function jump() {
            const target = window.parent.document.getElementById(anchor);
            if (!target && attempts++ < 180) {
                window.requestAnimationFrame(jump);
                return;
            }
            if (target) {
                target.scrollIntoView({block: 'start', behavior: 'instant'});
                target.focus({preventScroll: true});
            }
        }
        window.requestAnimationFrame(() => window.requestAnimationFrame(jump));
        </script>""",
        height=0, scrolling=False,
    )
