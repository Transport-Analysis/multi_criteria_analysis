import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import textwrap

from src import utils

# set up streamlit page
page_config = utils.setup_page()

# create header
st.header(page_config.page_header)
st.write(page_config.page_intro)

if st.button("Glossary", key=0):
    st.sidebar.subheader("Glossary")
    for glossary_item in page_config.glossary:
        st.sidebar.write(f'{glossary_item.name}: {glossary_item.definition}')

# import main image for page
utils.import_page_image()

# import project configs
import_proj_desc = utils.import_project_descriptions().ProjectDescription
criteria_list = utils.import_criteria_list().CriteriaList
nof_solutions = utils.import_nof_solutions().NOFsolutions

# Step 1 - import tool
st.subheader("1. File Import (Optional)")
with st.expander(
        'Import data from previously saved attempts (if applicable)',
        expanded=False
    ):

    if st.button("Help", key=1):
        st.sidebar.markdown("**Import Help**")
        st.sidebar.write(page_config.import_tool_help)

    uploaded_project = st.file_uploader(
        'Upload Saved Excel Project (Files downloaded from this website only)',
        type='xlsx'
    )

    if uploaded_project is not None:
        user_inputs = pd.read_excel(
            uploaded_project, sheet_name='user_inputs'
        )
        option_description = pd.read_excel(
            uploaded_project, sheet_name='option_description'
        )
        project_description = pd.read_excel(
            uploaded_project, sheet_name='project_description'
        )
        st.markdown('You uploaded a file successfully.')
    else:
        user_inputs = pd.DataFrame(columns=['Criterion', 'Ranks'])
        option_description = pd.DataFrame(
            columns=['Option', 'OptionDescription', 'Type']
        )
        project_description = pd.DataFrame(columns=['Category', 'Responses'])

    user_inputs.set_index('Criterion', inplace=True)

# Step 2 - Project Details
st.subheader("2. Project Details")
with st.expander("Project Description", expanded=False):

    if st.button("Help", key=2):
        st.sidebar.markdown("**Project Description Help**")
        st.sidebar.write(page_config.project_details_help)

    st.write(page_config.project_details_desc)

    input_responses = []
    categories_used = []
    options = [p.Options for p in import_proj_desc if p.Options is not None][0]

    if not project_description.empty:

        default = str(project_description.iloc[4, 4])
        for _, row in project_description.iterrows():
            if row.Category in categories_used:
                st.write('')
            else:
                categories_used.append(row.Category)
                st.write(f'{row.Category}')

            if row.hasnans:
                responses = st.text_input(
                    row.Question, value=row.Responses,
                    help="Press Enter to Apply"
                )
            else:
                responses = ','.join(
                    [x for x in st.multiselect(
                        label=project_description.iloc[4, 2],
                        options=options,
                        default=default
                    )]
                )

            input_responses.append(
                [row.Category, row.Question, row.Options, categories_used])

    else:

        for proj_desc in import_proj_desc:

            if proj_desc.Category in categories_used:
                st.write('')
            else:
                categories_used.append(proj_desc.Category)
                st.write(f'{proj_desc.Category}')

            if proj_desc.Options is None:
                responses = st.text_input(
                    proj_desc.Question, help='Press Enter to Apply'
                )
            else:
                responses = ','.join(
                    [x for x in st.multiselect(
                        proj_desc.Question, proj_desc.Options
                    )]
                )

            input_responses.append(
                [proj_desc.Category, proj_desc.Question,
                 proj_desc.Options, responses]
            )

    output_project_description = pd.DataFrame(
        input_responses,
        columns=['Category', 'Question', 'Options', 'Responses']
    )

# Step 3. Define Options
st.subheader("3. Define Options")
with st.expander("Define Options", expanded=False):

    if st.button("Help", key=3):
        st.sidebar.markdown("**Options Help**")
        st.sidebar.write(page_config.options_help)

    existing_options = []
    new_options = []

    # firstly read in user options already uploaded
    if not option_description.empty:

        st.write('Uploaded User Options')

        for _, row in option_description.iterrows():

            if row.Type == "User defined option":
                col1, col2 = st.columns(2)
                with col1:
                    option_name = st.text_input(
                        'User defined option', row.Option, key=row.Option
                    )
                with col2:
                    option_desc = st.text_input(
                        'User defined option comment',
                        row.OptionDescription,
                        key=row.OptionDescription
                    )
                option_type = 'User defined option'
            elif row.Type == 'Predefined option':
                option_name = row.Option
                option_desc = row.OptionDescription
                option_type = 'Predefined option'
            else:
                if row.Option == "Add new option":
                    pass
                else:
                    raise NotImplementedError(
                        'Only user defined and predefined options included')

            existing_options.append([option_name, option_desc, option_type])

    st.write('Select Your Own Options')
    # secondly allow for new options to be included
    i = len(
        [i for i in existing_options
         if existing_options[2] == 'User defined option']) + 1
    while True:
        col1, col2 = st.columns(2)
        with col1:
            new_option = st.text_input(
                f'What is option {i}?', 'Add new option', key=f'new_option{i}')
        with col2:
            new_option_comment = st.text_area(
                'Option description',
                'Add new option comment',
                key=f'new_option_comment{i}'
            )

        new_option_type = 'User defined option'
        if 'Add new option' in (new_option, new_option_comment):
            break
        else:
            new_options.append(
                [new_option, new_option_comment, new_option_type]
            )
            i += 1

    # finally load the pre-defined options
    st.write('Predefined Options')
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        st.write('Relevant Option?')
    with col2:
        st.write('NOS')
    with col3:
        st.write('')

    uploaded_nof_solutions = [
        n[0] for n in existing_options
        if existing_options[2] == 'Predefined option']

    nof_i = 0
    for nof_sol in nof_solutions:
        nof_i += 1
        col1, col2, col3 = st.columns([1, 1, 3])
        if nof_sol.Solution in uploaded_nof_solutions:

            with col1:
                checked = st.checkbox(
                    '',
                    key=f'NOFsolutions{nof_i}',
                    value=True
                )

            if checked:
                option_comment = [
                    n[1] for n in existing_options
                    if existing_options[0] == nof_sol.Solution][0]
                with col3:
                    ns = nof_sol.Solution
                    comment = st.text_input(
                        f'Additional comment on the solution: {ns}',
                        option_comment,
                        key=f'NOFsolutions_cm{ns}',
                        help=option_comment
                    )
        else:
            with col1:
                checked = st.checkbox(
                    '',
                    key=f'NOFsolutions{nof_i}'
                )

                if checked:
                    with col3:
                        ns = nof_sol.Solution
                        option_comment = st.text_input(
                            f'Additional comment on the solution: {ns}',
                            "Add comment",
                            key=f'NOFsolutions_cm{ns}',
                            help=nof_sol.Comment
                        )
                    new_options.append(
                        [nof_sol.Solution,
                         option_comment,
                         'Predefined option']
                    )

        with col2:
            st.markdown(nof_sol.Solution)

    all_options = existing_options + new_options
    output_option_description = pd.DataFrame(
        all_options, columns=['Option', 'OptionDescription', 'Type']
    )

# Step 4. Criteria ####
st.subheader("4. Criteria")
with st.expander("Criteria", expanded=False):

    if st.button("Help", key=4):
        st.sidebar.markdown("**Criteria Help**")
        st.sidebar.write(page_config.criteria_help)

    st.write(page_config.criteria_desc)

    criteria_df = pd.DataFrame(
        [[i.Category,
          i.Criterion,
          i.Indicator,
          i.Measure,
          i.NOS_mandatory] for i in criteria_list],
        columns=['Category',
                 'Criterion',
                 'Indicator',
                 'Measure',
                 'NOS Mandatory']
    ).drop(columns=['Measure'])

    criteria_df.index = np.arange(1, len(criteria_df) + 1)
    st.dataframe(criteria_df)

    new_criteria_df = criteria_df.copy()
    input_criteria_df = new_criteria_df[
        new_criteria_df.Criterion.apply(lambda x: x in user_inputs.index)]

    nos_flag = st.checkbox("Include all NOS Option's mandatory criteria")
    nos_defaults = new_criteria_df.loc[
        new_criteria_df['NOS Mandatory'] == True].index if nos_flag else ''
    try:
        if user_inputs.empty:
            selected_rows = st.multiselect(
                'Select rows:',
                new_criteria_df.index,
                default=[x for x in nos_defaults]
            )
        else:
            selected_rows = st.multiselect(
                'Select rows:', new_criteria_df.index,
                default=[
                    x for x in
                    list(set(nos_defaults) | set(input_criteria_df.index))]
            )
        selected_criteria = new_criteria_df.loc[
            selected_rows].sort_index()[['Category', 'Criterion']]

        st.write('Choose your own criteria')
        new_custom_criteria = []
        i = 1
        while True:
            col1, col2 = st.columns(2)
            with col1:
                new_criteria = st.text_input(
                    f'What is criteria {i}?',
                    'Add new criteria',
                    key=f'new_criteria{i}'
                )
            with col2:
                new_criteria_category = st.text_area(
                    'Criteria Category',
                    'Add new criteria category',
                    key=f'new_criteria_comment{i}'
                )
            new_custom_criteria.append([new_criteria_category, new_criteria])
            if 'Add new criteria' in (new_criteria, new_criteria_category):
                break
            else:
                i += 1

        if len(new_custom_criteria) > 0:
            new_custom_criteria_df = pd.DataFrame(
                new_custom_criteria,
                columns=['Category', 'Criterion']
            )

        all_criteria_used_df = pd.concat(
            [selected_criteria,
             new_custom_criteria_df]
        )
        all_criteria_used_df = all_criteria_used_df[
            all_criteria_used_df['Criterion'] != 'Add new criteria'].copy(
        ).reset_index(drop=True)
        all_criteria_used_df['Ranks'] = 0

        st.write('##### Selected Criteria')
        col11, col22 = st.columns([2, 1])
        for _, row in all_criteria_used_df.iterrows():
            with col11:
                st.write(row.Criterion)

    except:
        pass

st.subheader("5. Criteria Ranking and Option Scoring")
available_ranks = list(range(1, len(all_criteria_used_df) + 1))
with st.expander("Criteria Ranking & Scoring", expanded=False):

    if st.button("Help", key=5):
        st.sidebar.markdown("**Rankings Help**")
        st.sidebar.write(page_config.ranking_help)

    updated_user_inputs = []
    for i, row in all_criteria_used_df.iterrows():

        st.subheader(f'{row.Criterion}')

        label = 'Criteria Rank'
        index = available_ranks.index(
            row.Ranks) if row.Ranks in available_ranks else 0
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
        with col1:
            all_criteria_used_df.at[i, 'Ranks'] = st.selectbox(
                label, available_ranks, index, key=f'rank_{row.Criterion}'
            )

        num_options = max(1, len(output_option_description))
        num_rows_needed = int((num_options / 3))

        i = 1
        cols = st.columns(num_options)
        for _, option in output_option_description.iterrows():

            key = f'scores_{row.Criterion}_{option.Option}'
            if option.Option in user_inputs.columns:
                key_value = user_inputs[
                    user_inputs['Criterion'] == row.Criterion][option.Option]
            else:
                key_value = 3

            result = cols[i-1].select_slider(
                f'Score - option: {option.Option}',
                range(1, 6),
                key=key,
                value=key_value
            )

            updated_user_inputs.append([row.Criterion, option.Option, result])
            i += 1

        available_ranks = [
            x for x in available_ranks if x not in
            list(all_criteria_used_df.Ranks.unique())]

    updated_user_inputs = pd.DataFrame(
        updated_user_inputs,
        columns=['Criterion', 'Option', 'Value']
    )

    final_user_inputs = updated_user_inputs.pivot(
        index=['Criterion'],
        columns=['Option'],
        values='Value'
    ).reset_index()

    output_user_inputs = all_criteria_used_df.merge(
        final_user_inputs, on=['Criterion']
    ).drop(columns=['Category']).set_index(['Criterion'])

st.subheader("6. Results")
with st.expander("Results", expanded=False):

    if st.button("Help", key=6):
        st.sidebar.markdown("**Results Help**")
        st.sidebar.write(page_config.results_help)

    try:
        st.write(page_config.results_desc)
        st.write('Summary of Option Rating:')

        if not final_user_inputs.empty:
            fig, ax1 = plt.subplots(figsize=(20, 8))
            plt.subplots_adjust(bottom=0.2)
            plt.ylim(0, 6)
            sns.set_palette("colorblind")
            plot = sns.barplot(
                x='Criterion',
                y='Value',
                hue='Option',
                data=updated_user_inputs,
                ax=ax1
            )
            labels = [
                textwrap.fill(label.get_text(), 12)
                for label in plot.get_xticklabels()
            ]
            plot.set_xticklabels(labels)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

        rank_sums = [
            len(all_criteria_used_df) - x + 1
            for x in all_criteria_used_df.Ranks
        ]
        rank_sums_total = sum(rank_sums)
        rank_sums = [x / rank_sums_total for x in rank_sums]
        rank_sums = np.array(rank_sums)[:, np.newaxis]
        options = options = list(final_user_inputs.columns[1:])
        if len(final_user_inputs.columns) > 2 and len(final_user_inputs) > 1:
            user_scores = final_user_inputs.iloc[:, 1:].to_numpy()
            # to add the base case with score 3
            scores = np.c_[np.ones(len(user_scores))+2, user_scores]
            scores *= rank_sums
            st.write('Summary of Option Scoring:')
            scores_total = scores.sum(axis=0)
            overall_score_df = pd.DataFrame(scores_total).transpose()
            overall_score_df.columns = ['Base Case'] + options
            overall_score_df['title'] = 'Score'
            overall_score_df.set_index('title', inplace=True)
            overall_score_df = overall_score_df.transpose().sort_values(
                by='Score', ascending=False
            )
            overall_score_df = overall_score_df.style.format(
                subset=['Score'], formatter="{:.2}"
            )
            overall_score_df

            # Summary of Option Rankings ####
            st.write('Summary of Option Rankings:')
            tmp = (-scores_total).argsort()
            final_ranks = np.empty_like(tmp)
            final_ranks[tmp] = np.arange(len(scores_total))
            final_ranks += 1
            overall_rank_df = pd.DataFrame(final_ranks).transpose()
            overall_rank_df.columns = ['Base Case'] + options
            overall_rank_df['title'] = 'Rank'
            overall_rank_df.set_index('title', inplace=True)
            overall_rank_df = overall_rank_df.transpose().sort_values(
                by='Rank'
            )
            overall_rank_df

            # Best Option ####
            st.subheader('Best Option')
            checked = st.checkbox('Exclude Base Case?')
            st.write(
                f'Overall: {overall_rank_df.index[overall_rank_df["Rank"] == 1][0]}')

            scores_by_criteria = selected_criteria.copy().reset_index(
                drop=True
            )

            scores_df = pd.DataFrame(scores)
            scores_df.columns = ['Base Case'] + options
            scores_by_criteria_df = scores_df.join(
                scores_by_criteria
            ).drop(columns=['Criterion'])

            if checked:
                scores_by_criteria_df = scores_by_criteria_df.drop(
                    columns=['Base Case']
                )

            scores_by_criteria_df = pd.melt(
                scores_by_criteria_df,
                id_vars='Category'
            ).rename(columns={'variable': 'Option'})

            scores_by_criteria_df = scores_by_criteria_df.groupby(
                ['Category', 'Option']
            ).agg({'value': 'sum'}
                  ).reset_index().sort_values(
                      by=['Category', 'value'],
                      ascending=False
            )

            max_scores_by_criteria_df = scores_by_criteria_df.groupby(
                ['Category']).agg({'value': 'max'}).reset_index()

            best_scores_df = scores_by_criteria_df.merge(
                max_scores_by_criteria_df,
                on=['Category', 'value']
            )
            best_scores_df['Best Option'] = best_scores_df.groupby(
                ['Category']
            )['Option'].transform(lambda x: ', '.join(x))
            best_scores_df = best_scores_df.groupby(
                ['Category']
            ).agg({'Best Option': 'first'})
            st.write('Best option of each category')
            best_scores_df

            output_best_scores_df = scores_by_criteria_df.pivot(
                index=['Category'],
                columns=['Option'],
                values='value'
            )
            output_best_scores_df = output_best_scores_df.join(best_scores_df)

            # Functionality to Export Results ####
            # Download data
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:

                dfs_to_export = {
                    'project_description': output_project_description,
                    'option_description': output_option_description,
                    'scores_by_category': output_best_scores_df,
                    'overall_scores': overall_score_df,
                    'overall_rank': overall_rank_df,
                    'user_inputs': output_user_inputs
                }

                for sheet_name, df in dfs_to_export.items():
                    df.to_excel(writer, sheet_name)
                
                writer.save()

                st.download_button(
                    label='Download data to Excel',
                    data=buffer,
                    file_name="nof-mca-tool.xlsx",
                    mime="application/vnd.ms-excel"
                )

    except:
        pass

# Step 7. Sensitivity Analysis ####
st.subheader("7. Sensitivity Analysis")
with st.expander("Sensitivity Test", expanded=False):
    if st.button("Help", key=7):
        st.sidebar.markdown("**Sensitivity Test Help**")
        st.sidebar.write(page_config.sensitivity_test_help)

    st.write(page_config.sensitivity_test_desc)

    value = [-50, -25, 25, 50]
    input_value = []

    try:
        score = pd.DataFrame(user_scores)
        new_rank_sums = []
        cols = st.columns(4)
        for i in range(len(cols)):
            input = cols[i].number_input(
                'Change in Criteria Weighting Scenario' + str(i+1) + ' (in %)',
                min_value=-75,
                max_value=75,
                step=5,
                value=value[i],
                key=i
            )
            input_value.append(input)
            new_rank_sums.append(rank_sums*(1+input/100))

        # New Ranks
        scenario_results = []
        for scenario in range(0, len(new_rank_sums)):
            columns = []
            for r1 in output_user_inputs.Ranks:
                rows = []
                for r2 in output_user_inputs.Ranks:
                    if r1 == r2:
                        rank_input = new_rank_sums[scenario][r1-1]
                        rows.append(rank_input)
                    else:
                        rank_input = rank_sums[r2-1]*(
                            1-new_rank_sums[scenario][r1-1]
                        )/(1-rank_sums[r1-1])

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
            new_scoring[str(input_value[scenario])+' %'] = new_scoring.idxmax(
                axis=1
            )
            new_scoring = new_scoring[str(input_value[scenario])+' %']
            scenario_results.append(new_scoring)
        scenario_results = pd.DataFrame(scenario_results).transpose()
        scenario_results.index = selected_criteria["Criterion"]
        st.subheader('Sensitivity Analysis Summary')
        st.dataframe(scenario_results)
    except:
        pass
