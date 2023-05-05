import io
import matplotlib.pyplot as plt
from matplotlib.style import use
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
    st.sidebar.write(page_config.link_desc)
# import main image for page
utils.import_page_image()

# import project configs
import_proj_desc = utils.import_project_descriptions().ProjectDescription
criteria_list = utils.import_criteria_list().CriteriaList
nof_solutions = utils.import_nof_solutions().NOFsolutions

# Create empty dataframes
output_best_scores_df = pd.DataFrame()
overall_score_df = pd.DataFrame()
overall_rank_df = pd.DataFrame()

# Step 1 - import tool
st.subheader("1. File Import (Optional)")
with st.expander(
        'File Import',
        expanded=False
):
    if st.button("Help", key=1):
        st.sidebar.markdown("**File Import**")
        st.sidebar.write(page_config.import_tool_help)

    st.write('Import data from previously saved attempts (if applicable)')

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
        user_inputs = pd.DataFrame(columns=['Criterion', 'Category', 'Weights'])
        option_description = pd.DataFrame(
            columns=['Option', 'OptionDescription', 'Type']
        )
        project_description = pd.DataFrame(columns=['Category', 'Responses'])

    user_inputs.set_index('Criterion', inplace=True)

# Step 2 - Project Details
st.subheader("2. Project Details")
with st.expander("Project Details", expanded=False):
    if st.button("Help", key=2):
        st.sidebar.markdown("**Project Details**")
        st.sidebar.write(page_config.project_details_help)

    st.write(page_config.project_details_desc)

    input_responses = []
    categories_used = []

    if not project_description.empty:

        key_objs = project_description[
            project_description['Category'] == 'Key Objectives'
            ].copy().reset_index(drop=True)

        default_list = key_objs.loc[0, 'Responses'].split(',')

        s = key_objs.loc[0, 'Options']
        options_list = s.replace('[', '').replace(']', '').replace("'", "").split(',')
        options = [o.strip() for o in options_list]

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
                        default=default_list
                    )]
                )

            input_responses.append(
                [row.Category, row.Question, row.Options, responses])

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
        st.sidebar.markdown("**Define Options**")
        st.sidebar.write(page_config.options_help)

    existing_options = []
    new_options = []

    # firstly read in user options already uploaded
    if not option_description.empty:

        st.write('Uploaded User Options')

        i = 1
        for _, row in option_description.iterrows():

            if row.Type == "User defined option":
                col1, col2 = st.columns(2)
                with col1:
                    option_name = st.text_input(
                        'User defined option', row.Option,
                        key=f'{row.Option}_{i}'
                    )
                with col2:
                    option_desc = st.text_input(
                        'User defined option comment',
                        row.OptionDescription,
                        key=f'{row.OptionDescription}_{i}'
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
            i += 1

            existing_options.append([option_name, option_desc, option_type])

    st.write('Use this section to list the options being assessed.')
    # secondly allow for new options to be included
    i = len(
        [i for i in existing_options if i[2] == 'User defined option']) + 1
    while True:
        col1, col2 = st.columns(2)
        with col1:
            new_option = st.text_input(
                f'What is option {i}?', value='', help='Add new option', key=f'new_option{i}')
        with col2:
            new_option_comment = st.text_input(
                'Option description',
                value='', help='Add new option comment',
                key=f'new_option_comment{i}'
            )

        new_option_type = 'User defined option'
        if len(new_option) < 1:
            break
        else:
            new_options.append(
                [new_option, new_option_comment, new_option_type]
            )
            i += 1

    # finally load the pre-defined options
    st.write('Choose from predefined NOS options:')
    col1, col2, col3 = st.columns([1, 1, 3])

    uploaded_nof_solutions = [
        n[0] for n in existing_options if n[2] == 'Predefined option'
    ]

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
                    if n[0] == nof_sol.Solution][0]
                with col3:
                    ns = nof_sol.Solution
                    comment = st.text_input(
                        'Additional comment on the solution',
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
                            'Additional comment on the solution',
                            "Add description of solution",
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

# Step 4. Define Criteria
st.subheader("4. Define Criteria")
with st.expander("Define Criteria", expanded=False):
    if st.button("Help", key=4):
        st.sidebar.markdown("**Define Criteria**")
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
        new_criteria_df['NOS Mandatory'] == True].Criterion if nos_flag else ''
    try:
        if user_inputs.empty:
            selected_rows = st.multiselect(
                'Select criteria:',
                new_criteria_df.Criterion,
                default=[x for x in nos_defaults]
            )
            additional_criteria_df = pd.DataFrame()

        else:
            selected_rows = st.multiselect(
                'Select criteria:', new_criteria_df.Criterion,
                default=[
                    x for x in
                    list(set(nos_defaults) | set(input_criteria_df.Criterion))]
            )

            df = user_inputs.merge(input_criteria_df[['Criterion']], on='Criterion', how='left', indicator=True)

            additional_criteria_df = df[df['_merge'] == 'left_only'].copy()[['Criterion', 'Category']]

            st.write('Custom Criteria Added:')
            final_additional_criteria = []
            i = 1
            col1, col2 = st.columns(2)
            for _, row in additional_criteria_df.iterrows():
                with col1:
                    added_criteria = st.text_input(
                        label=f'What is criteria {i}?',
                        value=row.Criterion,
                        key=row.Criterion
                    )
                with col2:
                    added_category = st.text_input(
                        label='Criteria Category',
                        value=row.Category,
                        key=f'category_{row.Criterion}'
                    )
                i += 1
                final_additional_criteria.append([row.Criterion, row.Category])

            if len(final_additional_criteria) > 0:
                additional_criteria_df = pd.DataFrame(final_additional_criteria, columns=['Criterion', 'Category'])

        selected_criteria = new_criteria_df[new_criteria_df['Criterion'].isin(selected_rows)][["Category", "Criterion"]]
        selected_criteria = pd.concat([selected_criteria, additional_criteria_df])

        st.write('Choose Your Own Criteria:')
        new_custom_criteria = []
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

        if not user_inputs.empty:
            cri_weights = user_inputs['Weights'].to_dict()
            all_criteria_used_df['Weights'] = all_criteria_used_df['Criterion'].map(cri_weights).astype(float)
        else:
            all_criteria_used_df['Weights'] = 0.01
            all_criteria_used_df['Weights'] = all_criteria_used_df['Weights'].astype(float)

        st.write('##### Selected Criteria')
        col11, col22 = st.columns([2, 1])
        for _, row in all_criteria_used_df.iterrows():
            with col11:
                st.write(f'Criterion: {row.Criterion}, Category: {row.Category}')

    except:
        pass

# Step 5. Criteria Weights
st.subheader("5. Criteria Weights")
with st.expander("Criteria Weights", expanded=False):
    if st.button("Help", key=5):
        st.sidebar.markdown("**Criteria Weights**")
        st.sidebar.write(page_config.weight_help)
    crit_weights = []
    weights = []
    cols = st.columns(4)
    for i, row in all_criteria_used_df.iterrows():
        if i < 4:
            weight = cols[i].number_input(
                label=f'Criteria Weight - {row.Criterion}',
                value=row.Weights,
                min_value=0.01,
                max_value=1.00, help='Weights do not need to sum to 1.',
                key=f'Weight_{row.Criterion}')

        else:
            weight = cols[i - 4].number_input(
                label=f'Criteria Weight - {row.Criterion}',
                value=row.Weights,
                min_value=0.01,
                max_value=1.00, help='Weights do not need to sum to 1.',
                key=f'Weight_{row.Criterion}')
        weights.append(weight)
        crit_weights.append([row.Criterion, weight])

    weights_df = pd.DataFrame(crit_weights, columns=['Criterion', 'Weight'])

# Step 6. Scoring
st.subheader('6. Scoring')
with st.expander("Scoring", expanded=False):
    if st.button("Help", key=6):
        st.sidebar.markdown("**Scoring**")
        st.sidebar.write(page_config.scoring_help)

    st.write(page_config.scoring_desc)

    updated_user_inputs = []
    num_criteria = len(all_criteria_used_df)

    for i, row in all_criteria_used_df.iterrows():

        st.subheader(f'{row.Criterion}')

        num_options = max(1, len(output_option_description))
        num_rows_needed = int((num_options / 3))

        i = 1
        cols = st.columns(num_options)
        for _, option in output_option_description.iterrows():

            key = f'scores_{row.Criterion}_{option.Option}'
            if option.Option in user_inputs.columns and row.Criterion in user_inputs.index:
                key_value = user_inputs[
                    user_inputs.index == row.Criterion].loc[
                    row.Criterion, option.Option
                ]
            else:
                key_value = 3

            result = cols[i - 1].select_slider(
                f'Score - option: {option.Option}',
                range(1, 6),
                key=key,
                value=key_value
            )

            updated_user_inputs.append([row.Criterion, option.Option, result])
            i += 1

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
    ).set_index(['Criterion'])

# Step 7. Results
st.subheader("7. Results")
with st.expander("Results", expanded=False):
    if st.button("Help", key=7):
        st.sidebar.markdown("**Results**")
        st.sidebar.write(page_config.results_help)

    try:
        st.write(page_config.results_desc)
        st.write('Summary of option scores:')

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
            st.pyplot(fig)
            img = io.BytesIO()
            plt.savefig(img, format='png')

            st.download_button(
                label='Download Graph',
                data=img,
                file_name='mca_scores.png',
                mime='image/png'
            )
        if len(final_user_inputs.columns) > 2 and len(final_user_inputs) > 1:

            def adjust_weights_df(wgts_df):
                wgts_df['weights_total'] = wgts_df.Weight.sum()
                wgts_df['adj_weight'] = wgts_df['Weight'] / wgts_df['weights_total']
                return wgts_df.drop(columns=['Weight', 'weights_total'])


            adjusted_weights_df = adjust_weights_df(weights_df)
            raw_df = final_user_inputs.merge(adjusted_weights_df, on=['Criterion'])

            st.write('Final Criteria Weights used (balanced to total to 1.0):')
            final_weights_df = all_criteria_used_df.merge(
                adjusted_weights_df, on=['Criterion']
            ).drop(columns=['Weights']).rename(columns={'adj_weight': 'Weight'}).set_index(['Category', 'Criterion'])
            final_weights_df

            # add in Base Case
            raw_df['Base Case'] = 3
            option_cols = [x for x in raw_df.columns if x not in ['Criterion', 'adj_weight']]


            def get_weighted_scores(_raw_df, cols):
                df = _raw_df.copy()
                for col in cols:
                    df[col] = df[col] * df['adj_weight']
                return df.drop(columns=['adj_weight'])


            # get weighted scores
            wght_scores_df = get_weighted_scores(raw_df, option_cols)

            # get final scores and ranks
            st.write('Summary of final option scores and rank:')
            total_scores_df = wght_scores_df[option_cols].copy()
            final_df = pd.DataFrame(total_scores_df.sum(axis=0)).rename(columns={0: 'Score'})
            final_df['Rank'] = final_df['Score'].rank(method='min', ascending=False)
            final_df = final_df.sort_values(by=['Rank'])
            final_df

            overall_score_df = final_df[['Score']].copy()
            overall_rank_df = final_df[['Rank']].copy()

            # Best Option
            st.subheader('Best Option')
            checked = st.checkbox('Exclude Base Case?')
            st.write(
                f'Overall best option: {final_df.index[final_df["Rank"] == 1][0]}'
            )

            category_df = all_criteria_used_df[['Criterion', 'Category']].copy().set_index(['Criterion'])
            category_scores_df = wght_scores_df.set_index(['Criterion']).join(category_df)

            if checked:
                category_scores_df = category_scores_df.drop(columns=['Base Case'])

            scores_by_criteria_df = pd.melt(
                category_scores_df,
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

            # update weights
            fw = final_weights_df.reset_index().set_index(
                'Criterion')['Weight'].to_dict()
            output_user_inputs['Weights'] = output_user_inputs.index.to_series(
            ).map(fw)

    except:
        pass

# Step 8. Sensitivity Test
st.subheader("8. Sensitivity Test")
with st.expander("Sensitivity Test", expanded=False):
    if st.button("Help", key=8):
        st.sidebar.markdown("**Sensitivity Test**")
        st.sidebar.write(page_config.sensitivity_test_help)

    st.write(page_config.sensitivity_test_desc)
    adj_weights = []
    try:

        cols = st.columns(4)
        for i, row in output_user_inputs.reset_index().iterrows():
            if i < 4:
                adj_weight = cols[i].number_input(
                    label=f'Adjusted Weight for {row.Criterion}',
                    min_value=0.01,
                    max_value=1.00,
                    value=row.Weights,
                    key=f'adj_weight{row.Criterion}'
                )
            else:
                adj_weight = cols[i - 4].number_input(
                    label=f'Adjusted Weight for {row.Criterion}',
                    min_value=0.01,
                    max_value=1.00,
                    value=row.Weights,
                    key=f'adj_weight{row.Criterion}')
            adj_weights.append([row.Criterion, adj_weight])

        sens_weights_df = pd.DataFrame(adj_weights, columns=['Criterion', 'Weight'])

        adj_sens_weights_df = adjust_weights_df(sens_weights_df)

        st.write('Summary of final adjusted sensitivity weights used (balanced to total to 1.0):')
        adj_sens_df = adj_sens_weights_df.set_index(['Criterion']).rename(columns={'adj_weight': 'Sensitivity Weight'})
        adj_sens_df

        sens_df = final_user_inputs.merge(adj_sens_weights_df, on=['Criterion'])

        sens_weights_df = all_criteria_used_df.merge(
            adjusted_weights_df, on=['Criterion']
        ).drop(columns=['Weights']).rename(columns={'adj_weight': 'Weight'}).set_index(['Category', 'Criterion'])

        # add in Base Case
        sens_df['Base Case'] = 3
        option_cols = [x for x in sens_df.columns if x not in ['Criterion', 'adj_weight']]

        # get weighted scores
        sens_wght_scores_df = get_weighted_scores(sens_df, option_cols)

        # get final scores and ranks
        st.write('Summary of final option scores and rank: Sensitivity Test')
        sens_total_scores_df = sens_wght_scores_df[option_cols].copy()
        sens_final_df = pd.DataFrame(sens_total_scores_df.sum(axis=0)).rename(columns={0: 'Score'})
        sens_final_df['Rank'] = sens_final_df['Score'].rank(method='min', ascending=False)
        sens_final_df = sens_final_df.sort_values(by=['Rank'])
        sens_final_df

    except:
        pass

# File Export to Excel
st.subheader("File Export")
st.write("Export session data to Excel to save the results and resume your progress next time.")

buffer = io.BytesIO()
writer = pd.ExcelWriter(buffer, engine='xlsxwriter')

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

# writer.save()
writer.close()

st.download_button(
    label='Download data to Excel',
    data=buffer,
    file_name="nof-mca-tool.xlsx",
    mime="application/vnd.ms-excel"
)
