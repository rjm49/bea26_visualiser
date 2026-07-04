import json
import numpy as np
import pandas as pd
import panel as pn

pn.extension("tabulator")

ACCENT = "teal"

styles = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
}

nav = pn.pane.HTML("""<nav><p><a href="/items_summary">Items Summary</a></p><p><a href="/student_lookup">Student Lookup</a></p><p><a href="/students_summary">Students Summary</a></p></nav>"""
)

# Extract Data

# @pn.cache()  # only download data once
def get_data():
    df = pd.read_csv("data/vis_data.csv", sep="\t")
    return df.sort_values(by="created_epoch")

source_data = get_data()

# Transform Data

min_datetime = int(source_data["created_epoch"].min())
max_datetime = int(source_data["created_epoch"].max())
top_manufacturers = (
    source_data.groupby("public_user_id").public_user_id.unique().sort_values().iloc[-100:].index.to_list()
)


def get_unique_item_data():
    data = source_data[["public_prompt_id", "mean_GERs_human_final", "mean_CEFRs_human", "prompt_level"]].drop_duplicates()
    mean_ger_levels = [np.mean(json.loads(v)) for v in data.mean_GERs_human_final.values]
    data["mean_GER_level_human_final"] = mean_ger_levels
    return data

# Filters
get_unique_item_data()
uniq_item_df = pn.rx(get_unique_item_data())
count = uniq_item_df.rx.len()
fig_item_scatter = uniq_item_df.hvplot.scatter(
    title="CEFR over time",
    x = "mean_GER_level_human_final",
    y = "mean_CEFRs_human",
    value_label= "Value",
    color="prompt_level"
)

image = pn.pane.JPG("""https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fpbs.twimg.com%2Fmedia%2FFZ9shGuX0AANFRe%3Fformat%3Djpg%26name%3D4096x4096&f=1&nofb=1&ipt=8fca8354c8075c8a887830efc055b22efa78d98be539c59799bc91f286dad78a""")
fig_item_scatter = pn.pane.HoloViews(fig_item_scatter, sizing_mode="stretch_both", name="Item Scatter")

pn.template.FastListTemplate(
    title="WI Items Summary",
    sidebar=[image, nav],
    # main=[pn.Column(indicators, tabs, matplot,)],  #sizing_mode="stretch_both")],
    main=[ fig_item_scatter ],
    main_layout=None,
    accent=ACCENT,
).servable()

