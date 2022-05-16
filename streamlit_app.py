# Import dependencies
import streamlit as st
import yaml
import pandas as pd
import numpy as np
import time
import io
import xlsxwriter

# Page Config #
st.set_page_config(
    page_title = "MCA Tool",
    page_icon = "🚴"
    )

#Set the page max width
def _max_width_():
    max_width_str = f"max-width: 1000px;"
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
_max_width_()

# Import data from input files
for filename in ('inputs', 'variables'):
    with open('%s.yaml' % filename) as file:
        Inpt_lst = yaml.load(file, Loader=yaml.FullLoader)
        for key, dcts in Inpt_lst.items():
            for i, dct in enumerate(dcts):
                db = pd.DataFrame.from_dict(dct, orient='index').transpose()
                if i == 0:
                    globals()[key] = db
                else:
                    globals()[key]  = globals()[key] .append(db)
            globals()[key] .index = np.arange(1, len(globals()[key])+1)
           

#### Introduction ####
st.header("Smarter Solutions")
st.subheader("Multi-Criteria Analysis (MCA) Tool")
st.write('''This Smarter Solutions Multi-Criteria Analysis **(MCA)** Tool provides a clear line-of-sight across the Department of Transport and Main Roads' **(TMR)** infrastructure planning and investment process, providing assurance that the Network Optimisation Framework is embedded in our decision-making.
    The MCA Tool has been designed for use in selecting a preferred option, or ranking alternate options, where network optimisation solutions **(NOS)** are included within assessment processes. The MCA Tool applies a standardised consideration of NOS relative to large capital infrastructure, ensuring TMR is delivering the right infrastructure at the right time and aligning with government policy direction for investment as outlined in the Queensland Government's State Infrastructure Plan.
''')

#### Project Description ####
with st.expander("Project Description", expanded=True):
    if st.button("Help", key=1):
        st.sidebar.write("Help with Project Description")
    st.write('''The project must be clearly defined within the MCA to ensure that appropriate options are short-listed for evaluation and that the criteria selected for assessment reflect the nature of the service requirement or opportunity. Accordingly, the project should be defined in terms of:''')
    answers = []     
    for _, row in ProjectDescription.iterrows():
        st.write('%s:' % row.Category)
        if row.hasnans: 
            answers.append(st.text_input(row.Question))
        else:
            answers.append(','.join([x for x in st.multiselect(row.Question, row.Options)]))
    ProjectDescription['answers'] = answers
ProjectDescription = ProjectDescription[['Category', 'answers']]

#### NOF Options #### 
# TODO - new section to be added.
# Add a section that asks the General User to complete a preliminary review of the 
# application of each NOF option (Yes/No) in the Smarter Solutions Reference Guide 
# to determine which option should be included in the rest of the MCA process. 

#### Define Options #### 
with st.expander("Define Options", expanded=True):
    if st.button("Help", key=2):
        st.sidebar.write("Help with Define Options")
    options = []
    st.write('Clearly define the short-listed options identified to achieve the outcomes sought.')
    i = 1
    while True:
        col1, col2 = st.columns(2)
        with col1:
            options.append(st.text_input('What is option %i?' % i, 'Add new option'))
        with col2:
            options[-1] = (options[-1], st.text_area('Option %i description' % i, 'Add new option'))
        if 'Add new option' in options[-1]:
            break
        i += 1
    options = options[:-1]

#### Criteria ####
with st.expander("Criteria", expanded=True):
    if st.button("Help", key=3):
        st.sidebar.write("Help with Criteria")
    st.write('''As per the Smarter solutions -  Multi-Criteria Assessment Technical Note, various criteria are mandatory when considering an NOS in the evaluation process. Additional criteria relating to intersection delay, public transport patronage and freight should be selected where appropriate. ''')
    st.dataframe(CriteriaList)
    nos_flag = st.checkbox(''' Include NOS Option's criteria''')
    nos_defaults = CriteriaList.loc[CriteriaList['NOS mandatory'] == True].index if nos_flag else ''
    SelectedRows = st.multiselect('Select rows:', CriteriaList.index, default=[x for x in nos_defaults])
    SelectedCriteria = CriteriaList.loc[SelectedRows].sort_index()
    SelectedCriteria = SelectedCriteria.iloc[:, :2]
    st.write('### Selected Criteria', SelectedCriteria)

#### Weightings Ranking ####
#### Scoring ####
AvailableRanks = range(1,len(SelectedCriteria) + 1)
Ranks = []
Scores = []
st.write('Rank Criteria:')
for indx, row in SelectedCriteria.iterrows():
    with st.expander("Category: %s & criterion: %s" % (row.Category, row.Criterion)):
        Ranks.append(st.selectbox('Rank - criterion: %s' %  row.Criterion, AvailableRanks, key=indx))
        for inr, itm in enumerate(options):
            Scores.append(st.select_slider('Score - option: %s' %  itm[0], range(1,6), key=indx*100+inr, value=3))
    AvailableRanks = [x for x in AvailableRanks if x not in Ranks]

RankSums = [len(Ranks) - x + 1 for x in Ranks]
RankSums_ttl = sum(RankSums)
RankSums = [x / RankSums_ttl for x in RankSums]
RankSums = np.array(RankSums)[:, np.newaxis]
if len(Scores) > 0:
    Scores = np.array(Scores).reshape(len(Ranks), -1)
    Scores = np.c_[np.ones(len(Scores))+2, Scores]
    Scores *= RankSums
    st.write('Summary of Option Scoring:')
    ScoresTotal = Scores.sum(axis=0)
    OverallScore = pd.DataFrame(ScoresTotal)
    OverallScore = OverallScore.transpose()
    OverallScore.columns = ['Base Case'] + [y for x, y in options]
    OverallScore['title'] = 'Score'
    OverallScore.set_index('title', inplace=True)
    OverallScore = OverallScore.transpose()
    OverallScore

    # Summary of Option Rankings
    st.write('Summary of Option Rankings:')
    tmp = (-ScoresTotal).argsort()
    FinalRanks = np.empty_like(tmp)
    FinalRanks[tmp] = np.arange(len(ScoresTotal))
    FinalRanks += 1
    OverallRank = pd.DataFrame(FinalRanks)
    OverallRank = OverallRank.transpose()
    OverallRank.columns = ['Base Case'] + [y for x, y in options]
    OverallRank['title'] = 'Rank'
    OverallRank.set_index('title', inplace=True)
    OverallRank = OverallRank.transpose()
    OverallRank.sort_values(['Rank'])
    OverallRank

    # Best Option
    st.header('Best Option:')
    st.subheader('Overall: \n%s' % OverallRank.index[np.where(FinalRanks==1)][0])
    scores_by_criteria = SelectedCriteria.copy()
    for j, (_, y) in enumerate(options):
        scores_by_criteria['Score_%s' % y] = [Scores[i, j + 1] / Ranks[i] for i in range(len(Scores))] # j + 1 instead of j to exclude BASE
    scores_by_category = scores_by_criteria.groupby('Category').sum()[['Score_%s' % y for x, y in options]]
    scores_by_category.columns = [x[6:] for x in scores_by_category.columns]
    scores_by_category['Best Option'] = scores_by_category.T.idxmax()
    scores_by_category = scores_by_category
    st.subheader('Best option of each criterion:')
    st.write('Base Case is excluded')
    scores_by_category.iloc[:, -1:]

    #### Functionality to Export Results ####
    # Download data
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        for key in ('ProjectDescription', 'scores_by_category', 'OverallScore', 'OverallRank'):
            globals()[key].to_excel(writer, sheet_name=key)
        writer.save()

        st.download_button(
            label="Download data to Excel",
            data=buffer,
            file_name="nof-mca-tool.xlsx",
            mime="application/vnd.ms-excel"
        )

        
#### Sensitivities ####
# TODO
