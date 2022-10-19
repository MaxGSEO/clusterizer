import streamlit as st

from helpers import local_css

st.set_page_config(
    layout="wide",
    page_title="The Entities Swissknife",
    page_icon="https://cdn.shortpixel.ai/spai/q_lossy+ret_img+to_auto/https://studiomakoto.it/wp-content/uploads/2021/08/cropped-favicon-16x16-1-192x192.png",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None
    }
)
hide_st_style = """
                    <style>
                    footer {visibility: hidden;}
                    [title^='streamlit_lottie.streamlit_lottie'] {
                        margin-bottom: -35px;
                        margin-top: -90px;
                    }
                    </style>
                    """
st.markdown(hide_st_style, unsafe_allow_html=True)
local_css("data/tes.css")
st.write("# Welcome to Clustering app👋")

st.markdown(
    """
    put your message here.
    """
)
