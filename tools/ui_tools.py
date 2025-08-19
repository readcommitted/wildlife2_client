"""
ui_tools.py — Shared Streamlit UI Utilities
---------------------------------------------

Provides reusable components for Streamlit apps, including:

* Scrollable progress/status area for long-running tasks
* Persistent log stored in `st.session_state`
* Safe HTML rendering for formatted display

Intended for use across ingestion, species detection, and other
workflows requiring real-time user feedback.

Dependencies:
- Streamlit

"""

import streamlit as st

# Global area to render progress messages
progress_area = st.empty()


def update_status(message: str, limit: int = 15):
    """
    Appends a message to the persistent progress log and updates the display area.

    Args:
        message (str): Status message to add (supports basic HTML)
        limit (int): Maximum number of recent messages to display (default 15)
    """
    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []

    st.session_state.progress_log.append(message)

    # Render only the most recent `limit` messages in scrollable div
    html = (
        "<div style='height: 350px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px;'>"
        + "<br>".join(st.session_state.progress_log[-limit:])
        + "</div>"
    )
    progress_area.markdown(html, unsafe_allow_html=True)
