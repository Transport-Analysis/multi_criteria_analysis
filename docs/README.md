# NOF MCA App Documentation
#### 1. Code overview
This software provides a Multi-Criteria Analysis (MCA) tool that has been designed for use in selecting a preferred option, or ranking alternate options, where Network Optimisation Solutions (NOS) are included within assessment processes. The code has been implemented primarily in Streamlit (see: https://streamlit.io/) which is a Python framework for data-driven apps.

#### 2. Structure of the code and directories
The main code is stored in /streamlit_app.py. 

Data, images and other scripts referred to in this script are in the following sub-directories:
- assets: contain images used in the tool i.e the MCA process 
- data: contains all yaml files with information about criteria, solutions, page config and variables 
- src: contains the 'utils.py' script to define all functions used throughout the main scripts
- docs: contain an example of the Excel file users can upload on the MCA tool for a test

#### 3. Package Requirements
A /requirements.txt file has been provided within the root of the repository to outline the Python libraries required for the MCA tool. When hosted on Streamlit, these packages should be automatically installed.

#### 4. Excel file (download and upload session)
User inputs to the MCA toll web app can be exported by the user with the “Download data to Excel” button located at the bottom of the page. This Excel file can then be uploaded via “File Upload” to resume progress at another time. An example project Excel file (nof-mca-tool__ABC_Road_example.xlsx) has been included in the docs directory.

#### 5. MCA Tool user interface text editing
Most of the text in the MCA Tool’s user interface has been abstracted into configuration (config) files located in the ‘data’ folder so that they can be easily edited. These files are in yaml format. A summary of the yaml files is below:
- page_config.yaml - contains text in the main user interface and sidebarhelp menus
- variables.yaml - data for the project description section of the MCA tool
- nof_solutions.yaml - data about the pre-defined NOS
- criteria.yaml - data about the pre-defined criteria

#### 6. Further Streamlit information and documentation
More information relating to Streamlit (in general) can be found on the official website. Streamlit documentation can be accessed at https://docs.streamlit.io/

#### 7. Contact information
For more information on this MCA Tool, please contact: Transport_Analysis_Requests@tmr.qld.gov.au.
