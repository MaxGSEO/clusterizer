import base64
import json
import streamlit as st


def load_lotti_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


def add_logo():
    with open("data/TES logo.png", "rb") as image_file:
        b_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("data:image/png;base64,%s");
                background-repeat: no-repeat;
                background-size: 280px 180px;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 50px;
                font-size: 20px;
                position: relative;
                top: 100px;
            }
        </style>
        """ % b_string,
        unsafe_allow_html=True,
    )


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
