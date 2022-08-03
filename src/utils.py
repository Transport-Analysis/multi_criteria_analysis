import streamlit as st
import os

from PIL import Image
from pydantic import BaseModel
from typing import List, Optional
from yaml import load, SafeLoader


class GlossaryConfig(BaseModel):
    name: str
    definition: str


class PageConfig(BaseModel):
    page_header: str
    page_intro: str
    import_tool_help: str
    project_details_help: str
    project_details_desc: str
    options_help: str
    criteria_help: str
    criteria_desc: str
    ranking_help: str
    results_help: str
    results_desc: str
    sensitivity_test_help: str
    sensitivity_test_desc: str
    glossary: List[GlossaryConfig]


class ProjDescListConfig(BaseModel):
    Category: str
    Question: str
    Options: Optional[List[str]]


class ProjectDescList(BaseModel):
    ProjectDescription: List[ProjDescListConfig]


class NOFSolutionConfig(BaseModel):
    Solution: str
    Comment: str


class NOFSolutionsList(BaseModel):
    NOFsolutions: List[NOFSolutionConfig]


class CriteriaListConfig(BaseModel):
    Category: str
    Criterion: str
    Indicator: Optional[str]
    Measure: Optional[str]
    NOS_mandatory: bool


class InputsList(BaseModel):
    CriteriaList: List[CriteriaListConfig]


def set_max_width():
    """ Sets the maximum width of the page
    """
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .appview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def colour_multiselect():
    st.markdown(
        """
    <style>
    span[data-baseweb="tag"] {
      background-color: rgb(0, 97, 100) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def slider_colour():
    ColourMinMax = st.markdown(
        ''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
            background: rgb(1 1 1 / 0%); } </style>''', 
            unsafe_allow_html = True
    )
    
    Slider_Cursor = st.markdown(
        ''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{  
        background-color: rgb(120, 120, 120); box-shadow: rgb(120 120 120 / 20%) 0px 0px 0px 0.2rem;} </style>''', 
        unsafe_allow_html=True
    )
    
    Slider_Number = st.markdown(
        ''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
        { color: rgb(50, 82, 123); } </style>''', 
        unsafe_allow_html = True
    )  
    
    col = f'''<style> div.stSlider >
    div[data-baseweb = "slider"] > div > div{{
    background: linear-gradient(to right,rgb(219, 67, 37) 0%, 
                            rgb(219, 67, 37) 50%,
                            rgb(0, 97, 100) 50%,
                            rgb(0, 97, 100) 100%);
                            }} </style>'''
    st.markdown(col, unsafe_allow_html=True)


def import_page_image():
    image = Image.open(
        os.path.join(os.getcwd(), 'assets', 'mca_process.png')
    )
    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col2:
        st.image(image, caption='')


def load_config(fp):
    with open(fp) as f:
        yaml_contents = load(f, Loader=SafeLoader)

    return yaml_contents


def import_page_config():
    cfg = os.path.join(os.getcwd(), 'data', 'page_config.yaml')
    cfg_contents = load_config(cfg)
    return PageConfig(**cfg_contents)


def import_project_descriptions():
    cfg = os.path.join(os.getcwd(), 'data', 'variables.yaml')
    cfg_contents = load_config(cfg)
    return ProjectDescList(**cfg_contents)


def import_nof_solutions():
    cfg = os.path.join(os.getcwd(), 'data', 'nof_solutions.yaml')
    cfg_contents = load_config(cfg)
    return NOFSolutionsList(**cfg_contents)


def import_criteria_list():
    cfg = os.path.join(os.getcwd(), 'data', 'criteria.yaml')
    cfg_contents = load_config(cfg)
    return InputsList(**cfg_contents)


def setup_page():
    """Sets up streamlit page
    """
    st.set_page_config(page_title="MCA Tool", page_icon="??")
    set_max_width()
    colour_multiselect()
    slider_colour()

    return import_page_config()
