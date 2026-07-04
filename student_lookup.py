import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas

print(pn.__version__)
print(hvplot.__version__)

from data_transforms import epoch_correct, get_final_row_df, get_promptfinal_row_df, get_mean_evp_components, \
    get_mean_ge_components,get_rank_df, get_rankings, ger_filter_data, get_promptfinal_data, \
    get_final_data_from_prompt_final, get_this_user_promptfinal, calc_ger_delta, calc_cefr_delta

evp_lookup = OrderedDict({"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5, "UNK": 6})

pn.extension("tabulator")
# pn.extension()


with open('all_error_types_compressed.json') as file:
    all_error_types = json.load(file)

ACCENT = "teal"

styles = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
}

# Extract Data

edit_selector = "edits_human_final"

# @pn.cache()  # only download data once
def get_data():
    df = pd.read_csv("data/vis_data.csv", sep="\t")
    df = df.sort_values(by="created_epoch")
    return df[0:100]

source_data = get_data()
source_data = epoch_correct(source_data)

students = (
    source_data.groupby("public_user_id").public_user_id.unique().sort_values().index.to_list()
)

students = sorted(students)
sel_student = pn.widgets.Select(
    name="Student ID",
    value=students[0],
    options=students,
    description="The ID of the student",
)

# Transform Data
final_row_df = get_final_row_df(source_data)
promptfinal_row_df = pn.bind( get_promptfinal_row_df, puid=sel_student, source_data=source_data)
mean_evp_components_df = get_mean_evp_components(evp_lookup, source_data)
mean_ge_components_df = get_mean_ge_components(source_data, edit_selector)

def get_final_evp_data(puid, mean_evp_components_df, source_data, evp_lookup):
    evp = source_data[source_data.public_user_id==puid].EVP_counts.tail(1).values[0]
    evp = json.loads(evp)
    v_lev = [evp[ix] for ix in evp_lookup.values()]
    weighted_levels = [evp[evp_lookup[k]] * evp_lookup[k] for k in list(evp_lookup.keys())[0:-1]]
    mean_evp = np.mean(weighted_levels)
    evp_df = pd.DataFrame({"EVP_level":v_lev, "EVP_cat":evp_lookup.keys()})
    evp_df["EVP_mean"] = mean_evp_components_df["EVP_level"]
    return evp_df, mean_evp
evp_df, mean_evp = pn.rx(get_final_evp_data)(puid=sel_student, mean_evp_components_df=mean_evp_components_df, source_data=source_data, evp_lookup=evp_lookup)

rank_df, num_users = get_rank_df(final_row_df, evp_lookup)
ger_rank, cefr_rank, evp_rank, ranked_out_of = pn.rx(get_rankings)(rank_df=rank_df, public_user_id=sel_student)


def update_ge_bar_chart(public_user_id, mean_ge_components_df):
    data = source_data[(source_data.public_user_id == public_user_id)]
    edits = json.loads(data[edit_selector].values[-1])
    lt_edits = json.loads(data.edits_this_LT.values[-1])
    pred_edits = json.loads(data.predicted_GER.values[-1]) #[0:-1]
    x = all_error_types
    y = [e for e in edits]
    lt_edits = [e for e in lt_edits]
    pred_edits = [e for e in pred_edits]
    sel = pd.DataFrame({"ge_type": x, "ge_rate": y, "ge_LT":lt_edits, "ge_pred":pred_edits, "ge_mean":mean_ge_components_df.GE_level})
    return sel.hvplot.bar(x='ge_type', y=['ge_rate', "ge_pred", "ge_LT", "ge_mean"],
                          title=f'GERP for {public_user_id}', rot=90, autorange="y", value_label="Component GE Rates", color=["teal", "orange", "cyan", "#ddd"])

ge_bar_chart = pn.bind(update_ge_bar_chart, sel_student, mean_ge_components_df)
mean_ger_df = pn.rx(ger_filter_data)(public_user_id=sel_student, source_data=source_data, edit_selector=edit_selector)

def calc_delta_df():
    user_list = []
    for puid in mean_ger_df.public_user_id.unique():
        user_df = mean_ger_df[mean_ger_df.public_user_id == puid]
        L = len(user_df)
        st = user_df.head(1)
        ed = user_df.tail(1)
        ger_delta  = (float(ed.mean_ger) - float(st.mean_ger) ) /L
        cefr_delta = (float(ed.cefr) - float(st.cefr) ) /L
        evp_delta  = (float(ed.w_mean_evp) - float(st.w_mean_evp) ) /L
        user_list.append([puid, ger_delta, cefr_delta, evp_delta])
    out_df = pd.DataFrame(user_list, columns= ["public_user_id", "ger_delta", "cefr_delta", "evp_delta"])
    # out_df.to_csv("calc_delta_df.csv")
    return out_df
delta_df = pn.rx(calc_delta_df)


promptfinal_df = get_promptfinal_data(source_data, edit_selector)
final_df = get_final_data_from_prompt_final(promptfinal_df)
this_user_promptfinal_df = pn.rx(get_this_user_promptfinal)(public_user_id=sel_student, promptfinal_df=promptfinal_df)

_, this_user_ger_delta = pn.rx(calc_ger_delta)(ger_df=mean_ger_df)
_, this_user_cefr_delta = pn.rx(calc_cefr_delta)(ger_df=mean_ger_df)

count = mean_ger_df.rx.len()
max_human_cefr = mean_ger_df.cefr_numeric_human.max()
avg_human_cefr = mean_ger_df.cefr_numeric_human.mean()
avg_auto_cefr = mean_ger_df.cefr_numeric.mean()


fig_evp = evp_df.hvplot.bar(title="Current EVP Vocabulary Profile",
                            x='EVP_cat', y=["EVP_level", "EVP_mean"], rot=90,
                            color=["teal","#ddd"], value="EVP levels", ylim=(0,1))

fig_cefr_timeseries = mean_ger_df.hvplot(
    title="CEFR over time",
    x = "created_epoch",
    y = ["cefr_numeric", "predicted_CEFR", "cefr_numeric_human"],
    rot=90,
    ylabel="CEFR",
    xlabel="Epoch TS",
    color = ["cyan", "orange", "teal"],
    value_label="CEFR",
    legend=False,
    dynamic=True
)

fig_ger_timeseries = mean_ger_df.hvplot(
    title="Mean GE rate over time",
    x = "created_epoch",
    y=["mean_ger_level_human", "mean_ger_edits_LT", "ai_pred_ger_level"],

    rot=90,
    # ylabel="GE Rate",
    xlabel="Epoch TS",
    color = ["teal", "cyan", "orange"],
    # xlim=(min_datetime, max_datetime),
    # color=ACCENT,
    value_label="Mean GE Rate AI",
    dynamic=True
)
image = pn.pane.JPG("https://imgcdn.stablediffusionweb.com/2024/9/25/149e8daa-a515-4575-9051-8ea7a08010b4.jpg")
nav = pn.pane.HTML("""<nav><p><a href="/items_summary">Items Summary</a></p><p><a href="/student_lookup">Student Lookup</a></p><p><a href="/students_summary">Students Summary</a></p></nav>"""
)

indicators = pn.FlexBox(
    pn.indicators.Number(
        value=count, name="#Drafts", format="{value:,.0f}", styles=styles
    ),
    pn.indicators.Number(
        value=max_human_cefr,
        name="Max Human CEFR",
        format="{value:,.1f}",
        styles=styles,
    ),
    pn.indicators.Number(
        value=avg_human_cefr,
        name="Avg. Human CEFR",
        format="{value:,.1f}",
        styles=styles,
    ),
    pn.indicators.Number(
        value=avg_auto_cefr,
        name="Avg. Auto CEFR",
        format="{value:,.1f}",
        styles=styles,
    ),
    pn.indicators.Number(
        value=mean_evp,
        name="Avg. EVP",
        format="{value:,.1f}",
        styles=styles,
    ),

    pn.indicators.Number(
        value=this_user_ger_delta,
        name="дGER (10k/draft)",
        format="{value:,.4f}",
        styles=styles,
    ),

        pn.indicators.Number(
        value=this_user_cefr_delta,
        name="дCEFR (per draft)",
        format="{value:,.4f}",
        styles=styles,
    ),

    pn.indicators.Number(
        value=ger_rank,
        name="Rank GER",
        format="{value:,.0f}",
        styles=styles,
    ),

    pn.indicators.Number(
        value=cefr_rank,
        name="Rank CEFR",
        format="{value:,.0f}",
        styles=styles,
    ),
    pn.indicators.Number(
        value=evp_rank,
        name="Rank EVP",
        format="{value:,.0f}",
        styles=styles,
    ),
)

fig_cefr = pn.pane.HoloViews(fig_cefr_timeseries, name="CEFR Time series", sizing_mode="stretch_both")
fig_ger = pn.pane.HoloViews(fig_ger_timeseries, name="Mean GER Time series", sizing_mode="stretch_both")
ge_bar_chart = pn.pane.HoloViews(ge_bar_chart, sizing_mode="stretch_both", name="Plot2")
fig_evp = pn.pane.HoloViews(fig_evp, sizing_mode="stretch_both", name="evp_plot")
series_stack = pn.GridSpec(ncols=6)
series_stack[0,0:3] = fig_cefr
series_stack[0,3:6] = fig_ger
bar_stack = pn.GridSpec(ncols=6)
bar_stack[0,0:4] = ge_bar_chart
bar_stack[0,4:6] = fig_evp
pn.template.FastListTemplate(
    title="WI Student Dashboard",
    sidebar=[image, sel_student, nav],
    main=[ pn.Column(indicators, series_stack, bar_stack)],
    main_layout=None,
    accent=ACCENT,
).servable()

# panel serve student_lookup.py --dev