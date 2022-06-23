# Import dependencies
import streamlit as st
import yaml
import pandas as pd
import numpy as np
import time
import io
import xlsxwriter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


# Page Config #
st.set_page_config(
    page_title = "MCA Tool",
    page_icon = "??"
    )

#Set the page max width
def _max_width_():
    max_width_str = f"max-width: 1200px;"
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

#Change colour of multiselect
def _colour_multiselect_():
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
_colour_multiselect_()

#Define Slider Colour
def slider_colour(slider):
    ColourMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)
    Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{  
    background-color: rgb(120, 120, 120); box-shadow: rgb(120 120 120 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)
    Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
    { color: rgb(50, 82, 123); } </style>''', unsafe_allow_html = True)   
    col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
    background: linear-gradient(to right,rgb(219, 67, 37) 0%, 
                            rgb(219, 67, 37) 50%, 
                            rgb(0, 97, 100) 50%, 
                            rgb(0, 97, 100) 100%);
                            }} </style>'''
    ColourSlider = st.markdown(col, unsafe_allow_html = True)
    return ColourMinMax, Slider_Cursor, Slider_Number, ColourSlider
    
# Import data from input files
for filename in ('inputs', 'variables', 'NOF_solutions'):
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
criteria_category = dict(zip(CriteriaList.Criterion, CriteriaList.Category))


#### Introduction ####
st.header("Smarter Solutions: Multi-Criteria Analysis (MCA) Tool")
st.write('''This MCA tool has been designed for use in selecting a preferred option, or ranking alternate options, where Network Optimisation Solutions (NOS) are included within assessment processes.
The MCA Tool applies a standardised consideration of NOS relative to large capital infrastructure, ensuring TMR is delivering the right infrastructure at the right time and aligning with government policy direction for investment as outlined in the Queensland Government's State Infrastructure Plan.
''')
if st.button("Glossary", key=0):
        st.sidebar.subheader("Glossary")
        st.sidebar.write("NOS: Network Optimisation Solution")
        st.sidebar.write("Option: Choice of NOS")
        st.sidebar.write("Criteria: Used to determine the best option")
        st.sidebar.write("Weightings: Assigned based on the rank of criteria")
        st.sidebar.write("Base case: Do nothing scenario")
                          
### Import Diagram        
from PIL import Image
image = Image.open('assets/diagram.JPG')
col1,col2,col3 = st.columns([1,5,0.2])
with col2: 
    st.image(image, caption='')
    
### Import Tool
with st.expander('(Optional) Import data from previously saved Excel file:', expanded=False):
    if st.button("Help", key=1):
            st.sidebar.write("Help with Import Tool")
    uploaded_project = st.file_uploader('Upload Saved Excel Project (Files downloaded from this website only)',type='xlsx')
    if uploaded_project is not None:
        UserInputs = pd.read_excel (uploaded_project, sheet_name='UserInputs')
        OptionDescription = pd.read_excel (uploaded_project, sheet_name='option_description')
        st.markdown('You uploaded a file successfully.')
    else:
        UserInputs = pd.DataFrame(columns=['Criterion','Ranks'])
        OptionDescription = pd.DataFrame(columns=['Option','OptionDescription','Type'])
    UserInputs.set_index('Criterion', inplace=True)
    
#### Project Description ####
with st.expander("Project Description", expanded=False):
    if st.button("Help", key=2):
        st.sidebar.write("Help with Project Description")
    st.write('''The project must be clearly defined within the MCA to ensure that appropriate options are short-listed for evaluation and that the criteria selected for assessment reflect the nature of the service requirement or opportunity. Accordingly, the project should be defined in terms of:''')
    answers = [] 
    category_used =[]    
    for _, row in ProjectDescription.iterrows():
        if row.Category in category_used:
            st.write('')
        else:
            category_used.append(row.Category)
            st.write('%s:' % row.Category)
        if row.hasnans: 
            answers.append(st.text_input(row.Question, help="Press Enter to Apply"))
        else:
            answers.append(','.join([x for x in st.multiselect(row.Question, row.Options)]))
    ProjectDescription['answers'] = answers
ProjectDescription = ProjectDescription[['Category', 'answers']]

#### NOF Options #### 
# New section that asks the General User to complete a preliminary review of the 
# application of each NOF option (Yes/No) in the Smarter Solutions Reference Guide 
# to determine which option should be included in the rest of the MCA process. 
with st.expander("Define Options", expanded=False):
    if st.button("Help", key=3):
        st.sidebar.write("Help with Define Options")
    #st.write('Choose from NOF options bla bla.')
    st.write('Choose Your Own Options')
    option_description = {'Option':[], 'OptionDescription':[], 'Type':[]}
    existing_option = []
    new_comment = {}
    for _,row in OptionDescription.iterrows():
        if row.Type == "User defined option" and row.Option != "Add new option":  
            col1, col2 = st.columns(2)
            with col1:
                existing_option.append(st.text_input('User defined option', row.Option, key = row.Option))
            with col2:
                st.text_input('User defined option comment', row.OptionDescription, key = row.OptionDescription)
        elif row.Type == "Predefined option":
            NOFsolutions.loc[NOFsolutions['Solution'] == row.Option, 'Comment'] = row.OptionDescription
        else:
            pass

    if not existing_option: 
        i = 1
    else:
        i = len(existing_option)
    while True:       
        col1, col2 = st.columns(2)
        with col1:
            new_option = st.text_input('What is option %i?' % i, 'Add new option', key='new_option%i' % i)
            option_description['Option'].append(new_option)
        with col2:
            new_option_comment = st.text_area('Option description', 'Add new option comment', key='new_option_comment%i' % i)
            option_description['OptionDescription'].append(new_option_comment)
            option_description['Type'].append('User defined option')
        if 'Add new option' in (new_option, new_option_comment):
            break
        else:
            UserInputs[new_option] = [3] * len(UserInputs)
            i+= 1
    
    st.write('Predefined Options')
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        st.write('Relevant Option?')
    with col2:
        st.write('NOS')
    with col3:
        st.write('')
    
    
    Updated_Input = []
    for itr, option in NOFsolutions.iterrows():                    
        col1, col2, col3 = st.columns([1,1,3])
        if option.Solution in UserInputs.columns:
            with col1:
                chck = st.checkbox('', key='NOFsolutions_ch%s' % itr, value=True)
            if chck:    
                with col3:
                    Comment = st.text_input('Additional comment on the solution: %s' % option.Solution, option.Comment, key='NOFsolutions_cm%s' % option.Solution, help=option.Comment)

            else:
                Updated_Input.append(option.Solution)

        else:
            with col1:
                chck = st.checkbox('', key='NOFsolutions_ch%s' % itr)
                if chck:
                    with col3:
                        Comment = st.text_input('Additional comment on the solution: %s' % option.Solution, "Add more comment", key='NOFsolutions_cm%s' % option.Solution, help=option.Comment)
                        option_description['Option'].append(option.Solution)
                        option_description['OptionDescription'].append(Comment)
                        option_description['Type'].append('Predefined option')
                        UserInputs[option.Solution] = [3] * len(UserInputs)
        with col2:
            st.markdown(option.Solution)

    option_description = pd.DataFrame(option_description)

#### Criteria ####
with st.expander("Criteria", expanded=False):
    if st.button("Help", key=4):
        st.sidebar.write("Help with Criteria")
    st.write('''Various criteria are mandatory when considering an NOS in the evaluation process. Additional criteria relating to intersection delay, public transport patronage and freight should be selected where appropriate. ''')
    criteria=CriteriaList.drop(columns="Measure")
    st.dataframe(criteria)
    NewCriteria = pd.DataFrame(criteria.copy())    
    InputCriteria = NewCriteria[NewCriteria.Criterion.apply(lambda x: x in UserInputs.index)]
    nos_flag = st.checkbox(''' Include all NOS Option's mandatory criteria''')
    nos_defaults = NewCriteria.loc[NewCriteria['NOS mandatory'] == True].index if nos_flag else '' 
    if UserInputs.empty:
        SelectedRows = st.multiselect('Select rows:', NewCriteria.index, default=[x for x in nos_defaults])
    else:
        SelectedRows = st.multiselect('Select rows:', NewCriteria.index, default=[x for x in list(set(nos_defaults) | set(InputCriteria.index))])
    SelectedCriteria = NewCriteria.loc[SelectedRows].sort_index()
    st.write('Choose your own criteria')
    input_criteria = {"Category":[],"Criterion":[]}
    i = 1 
    while True:       
        col1, col2 = st.columns(2)
        with col1:
            new_criteria = st.text_input('What is criteria %i?' % i, 'Add new criteria', key='new_criteria%i' % i)
            input_criteria["Criterion"].append(new_criteria)
        with col2:
            new_criteria_comment = st.text_area('Criteria Category', 'Add new criteria category', key='new_criteria_comment%i' % i)
            input_criteria["Category"].append(new_criteria_comment)
        if 'Add new criteria' in (new_criteria, new_criteria_comment):
            break
        else:
            i+= 1
    input_criteria = pd.DataFrame(input_criteria)
    SelectedCriteria = SelectedCriteria.append(input_criteria, ignore_index=True)
    SelectedCriteria = SelectedCriteria[SelectedCriteria["Criterion"]!="Add new criteria"]
    
    if not SelectedCriteria[SelectedCriteria.Criterion.apply(lambda x: x not in UserInputs.index)].empty:
        for new_criterion in SelectedCriteria[SelectedCriteria.Criterion.apply(lambda x: x not in UserInputs.index)].iloc[:, 1]:
            UserInputs.loc[new_criterion] = [len(UserInputs) + 1] + [3] * (len(UserInputs.columns)-1) 
    #SelectedCriteria = CriteriaList.copy()
    SelectedCriteria = SelectedCriteria[SelectedCriteria.Criterion.apply(lambda x: x in UserInputs.index)]

    st.write('##### Selected Criteria')
    col11,col22 = st.columns([2,1])
    for Criterion, row in UserInputs.iterrows():
        with col11:
            st.write(Criterion)
            
#### Weightings Ranking ####
#### Scoring ####
AvailableRanks = list(range(1,len(UserInputs) + 1))
for Criterion, row in UserInputs.iterrows():
    st.write("Criterion: %s" % Criterion)
    label = 'Rank - criterion: %s' % Criterion
    index = AvailableRanks.index(row.Ranks) if row.Ranks in AvailableRanks else 0
    col1,col2,col3,col4 = st.columns([1.5,1,1,1])
    with col1:         
        UserInputs.at[Criterion, 'Ranks'] = st.selectbox(label, AvailableRanks, index, key='rank_%s' % Criterion)
    col11,col22,col33 = st.columns(3)
    i=1
    for OptionName in UserInputs.columns[i:][~UserInputs.columns[i:].isin(Updated_Input)]:
        value = UserInputs.at[Criterion, OptionName]
        key = 'scores_%s_%s' % (Criterion, OptionName)
        if i==1 or i%3==1:
            with col11:
                UserInputs.at[Criterion, OptionName] = st.select_slider('Score - option: %s' %  OptionName, range(1,6), key=key, value=value)
        elif i==2 or i%3==2:
            with col22:
                UserInputs.at[Criterion, OptionName] = st.select_slider('Score - option: %s' %  OptionName, range(1,6), key=key, value=value
        else:
            with col33:
                UserInputs.at[Criterion, OptionName] = st.select_slider('Score - option: %s' %  OptionName, range(1,6), key=key, value=value)                 
        c1, c2, c3, c4 = slider_colour(value)
        i+=1
    used = int(np.where(UserInputs.index.to_numpy() == Criterion)[0])
    AvailableRanks = [x for x in AvailableRanks if x not in list(UserInputs.Ranks.to_numpy())[:used+1]]
    
#Summary of Option Rating
with st.expander("Summary of Results", expanded=False):
    try:
        st.write('This section provides a summary of the scoring and ranking per criteria and options chosen')
        st.write('Summary of Option Rating:')
        if not UserInputs.empty:
            fig, ax1 = plt.subplots(figsize=(20,8)) 
            plt.subplots_adjust(bottom=0.2)
            plt.ylim(0, 6)
            select_input = UserInputs.loc[:,UserInputs.columns!="Ranks"].reset_index().melt(id_vars="Criterion").rename(columns={"variable":"Option", "value":"Score"})
            sns.set_palette("colorblind")
            plot = sns.barplot(x='Criterion', y='Score', hue='Option', data=select_input, ax=ax1)
            labels = [textwrap.fill(label.get_text(), 12) for label in plot.get_xticklabels()]
            plot.set_xticklabels(labels)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", width = 12)
            st.image(buf)

        RankSums = [len(UserInputs) - x + 1 for x in UserInputs.Ranks]
        RankSums_ttl = sum(RankSums)
        RankSums = [x / RankSums_ttl for x in RankSums]
        RankSums = np.array(RankSums)[:, np.newaxis]
        options = list(UserInputs.columns[1:])
        if len(UserInputs.columns) > 2 and len(UserInputs) > 1:
            UserScores = UserInputs.iloc[:, 1:].to_numpy()
            # to add the base case with score 3
            Scores = np.c_[np.ones(len(UserScores))+2, UserScores]
            Scores *= RankSums
            st.write('Summary of Option Scoring:')
            ScoresTotal = Scores.sum(axis=0)       
            OverallScore = pd.DataFrame(ScoresTotal)
            OverallScore = OverallScore.transpose()
            OverallScore.columns = ['Base Case'] + options
            OverallScore['title'] = 'Score'
            OverallScore.set_index('title', inplace=True)
            OverallScore = OverallScore.transpose().sort_values(by='Score')
            OverallScore = OverallScore.style.format(subset=['Score'], formatter="{:.2}")
            OverallScore

            # Summary of Option Rankings
            st.write('Summary of Option Rankings:')
            tmp = (-ScoresTotal).argsort()
            FinalRanks = np.empty_like(tmp)
            FinalRanks[tmp] = np.arange(len(ScoresTotal))
            FinalRanks += 1
            OverallRank = pd.DataFrame(FinalRanks)
            OverallRank = OverallRank.transpose()
            OverallRank.columns = ['Base Case'] + options
            OverallRank['title'] = 'Rank'
            OverallRank.set_index('title', inplace=True)
            OverallRank = OverallRank.transpose().sort_values(by='Rank')
            OverallRank.sort_values(['Rank'])
            OverallRank

            # Best Option
            st.header('Best Option:')
            st.subheader('Overall: \n%s' % OverallRank.index[np.where(FinalRanks==1)][0])
            scores_by_criteria = SelectedCriteria.copy()
            for j, y in enumerate(options):
                scores_by_criteria['Score_%s' % y] = [Scores[i, j + 1] / UserInputs.Ranks[i] for i in range(len(Scores))] # j + 1 instead of j to exclude BASE
            scores_by_criteria['Category'] = [criteria_category[x] for x in UserInputs.index]
            scores_by_category = scores_by_criteria.groupby('Category').sum()[['Score_%s' % y for y in options]]
            scores_by_category.columns = [x[6:] for x in scores_by_category.columns]
            scores_by_category['Best Option'] = scores_by_category.T.idxmax()
            scores_by_category = scores_by_category
            st.subheader('Best option of each category:')
            st.write('Base Case is excluded')
            scores_by_category.iloc[:, -1:]
            
            #### Functionality to Export Results ####
            # Download data
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                for key in ('ProjectDescription', 'option_description', 'scores_by_category', 'OverallScore', 'OverallRank', 'UserInputs'):
                    globals()[key].to_excel(writer, sheet_name=key)
                writer.save()

                st.download_button(
                    label="Download data to Excel",
                    data=buffer,
                    file_name="nof-mca-tool.xlsx",
                    mime="application/vnd.ms-excel"
                )
    except:
        pass

#### Sensitivities ####
with st.expander("Sensitivity Test", expanded=False):
    if st.button("Help", key=5):
        st.sidebar.write("Help with Sensitivity Testing")
    st.write("Sensitivity Testing is done to check whether the chosen option still remains the best after adjusting criteria weightings")
    value = [-50,-25,25,50]
    input_value =[]
    try:
        score = pd.DataFrame(UserScores)
        new_rank_sums = []
        for i in range(1,5):
            input= st.select_slider('Change in Criteria Weighting Scenario ' + str(i) + ' (in %)',options=range(-75,80,5), value=value[i-1], key=i)
            slider_colour(input)
            input_value.append(input)
            new_rank_sums.append(RankSums*(1+input/100))
        #New Ranks
        scenario_results = []
        for scenario in range(0,len(new_rank_sums)):
            columns =[]
            for r1 in UserInputs.Ranks:
                rows =[]
                for r2 in UserInputs.Ranks:
                    if r1==r2:
                        rank_input = new_rank_sums[scenario][r1-1]
                        rows.append(rank_input)
                    else:
                        rank_input = RankSums[r2-1]*(1-new_rank_sums[scenario][r1-1])/(1-RankSums[r1-1])
                        rows.append(rank_input)
                columns.append(rows)
            df = pd.DataFrame(columns)
            new_scoring = []
            for option in score:
                new_score = (score[option]*df).transpose().sum().astype(float)
                new_scoring.append(new_score)
            new_scoring = pd.DataFrame(new_scoring).transpose()
            new_scoring.columns = options
            new_scoring["Base Case"] = 3        
            new_scoring[str(input_value[scenario])+' %'] = new_scoring.idxmax(axis=1)
            new_scoring = new_scoring[str(input_value[scenario])+' %']
            scenario_results.append(new_scoring)
        scenario_results = pd.DataFrame(scenario_results).transpose()    
        scenario_results.index = SelectedCriteria["Criterion"]
        st.header('Summary of Sensitivity Test')
        st.write(scenario_results)
    except:
        pass
