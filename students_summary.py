import json
import numpy as np
import pandas as pd
import panel as pn

pn.extension("tabulator")

with open('all_error_types_compressed.json') as file:
    all_error_types = json.load(file)

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


def get_unique_student_data():
    outdata_list = []
    data = source_data[
        ["public_user_id", "public_prompt_id", "edits_human_final", "cefr_numeric_human", "prompt_level"]]
    #the above line should get a single row for each user-prompt pair, this representing the final human-given scores of each user for each prompt
    for puid in data.public_user_id.unique():
        pudf = data[data.public_user_id == puid]
        # mean_ger_levels = np.mean([json.loads( s ) for s in pudf.mean_GERs_human_final.values], axis=-1)
        mean_ger_levels = np.mean( json.loads( pudf.edits_human_final.values[-1] ))
        mean_cefr_level = pudf.cefr_numeric_human.values[-1]
        prompt_level = pudf.prompt_level.values[0]
        outdata_list.append([puid, mean_ger_levels, mean_cefr_level, prompt_level])

    outdata = pd.DataFrame(outdata_list, columns=["uid","mean_GERs","mean_CEFRs","prompt_level"])
    return outdata

def get_unique_user_final_delta_data():
    outdata_list = []
    data = source_data[
        ["public_user_id", "public_prompt_id", "edits_human_final", "cefr_numeric_human", "prompt_level"]]
    #the above line should get a single row for each user-prompt pair, this representing the final human-given scores of each user for each prompt
    for puid in data.public_user_id.unique():
        pudf = data[data.public_user_id == puid]
        lenp = len(pudf)
        pudf = pd.concat([pudf.head(1), pudf.tail(1)])
        mean_ger_levels = np.array([json.loads( s ) for s in pudf.edits_human_final.values])
        overall_ger_delta = (np.mean( mean_ger_levels[-1] - mean_ger_levels[0] )) / lenp
        overall_cefr_delta = (pudf.cefr_numeric_human.values[-1] - pudf.cefr_numeric_human.values[0]) / lenp
        prompt_level = pudf.prompt_level.values[0]
        outdata_list.append([puid, overall_ger_delta, overall_cefr_delta, prompt_level])

    outdata = pd.DataFrame(outdata_list, columns=["uid","overall_ger_delta","overall_cefr_delta","prompt_level"])
    return outdata

def get_componentwise_ger_delta_data():
    outdata_list = []
    data = source_data[
        ["public_user_id", "public_prompt_id", "edits_human_final", "cefr_numeric_human", "prompt_level"]]#.drop_duplicates()
    #the above line should get a single row for each user-prompt pair, this representing the final human-given scores of each user for each prompt
    ger_deltas_list = []
    cefr_deltas_list= []
    for puid in data.public_user_id.unique():
        pudf = data[data.public_user_id == puid]
        len_p = len(pudf)
        pudf = pd.concat([pudf.head(1), pudf.tail(1)])
        mean_ger_levels = np.array([json.loads( s ) for s in pudf.edits_human_final.values])
        overall_ger_delta = (mean_ger_levels[-1] - mean_ger_levels[0]) / len_p
        ger_deltas_list.append(overall_ger_delta)
        overall_cefr_delta = (pudf.cefr_numeric_human.values[-1] - pudf.cefr_numeric_human.values[0]) / len_p
        cefr_deltas_list.append(overall_cefr_delta)
        # prompt_level = pudf.prompt_level.values[0]
    overall_ger_delta = np.mean(ger_deltas_list, axis=0)
    # overall_cefr_delta = np.mean(cefr_deltas_list, axis=0)
    x = all_error_types
    # outdata_list = [[x, overall_ger_delta, overall_cefr_delta]]
    outdata = pd.DataFrame({"ge_type": x, "component_ger_delta": overall_ger_delta,})
    return outdata


def get_typical_final_GER():
    outdata_list = []
    data = source_data[
        ["public_user_id", "public_prompt_id", "edits_human_final", "cefr_numeric_human", "prompt_level"]]#.drop_duplicates()
    #the above line should get a single row for each user-prompt pair, this representing the final human-given scores of each user for each prompt
    ger_deltas_list = []
    cefr_deltas_list= []
    for puid in data.public_user_id.unique():
        pudf = data[data.public_user_id == puid]
        len_p = len(pudf)
        pudf = pudf.tail(1)
        mean_ger_levels = np.array([json.loads( s ) for s in pudf.edits_human_final.values])
        # overall_ger_delta = (mean_ger_levels[-1] - mean_ger_levels[0]) / len_p
        ger_deltas_list.append(mean_ger_levels)
        # overall_cefr_delta = (pudf.cefr_numeric_human.values[-1] - pudf.cefr_numeric_human.values[0]) / len_p
        # cefr_deltas_list.append(overall_cefr_delta)
        # prompt_level = pudf.prompt_level.values[0]
    overall_ger_delta = np.mean(ger_deltas_list, axis=0).ravel()
    # overall_cefr_delta = np.mean(cefr_deltas_list, axis=0)
    x = all_error_types
    # outdata_list = [[x, overall_ger_delta, overall_cefr_delta]]
    outdata = pd.DataFrame({"ge_type": x, "component_ger_delta": overall_ger_delta,})
    return outdata

def get_arrow_scatter_data():
    outdata_list = []
    data = source_data[
        ["public_user_id", "uix", "public_prompt_id", "edits_human_final", "cefr_numeric_human", "prompt_level"]]
    #the above line should get a single row for each user-prompt pair, this representing the final human-given scores of each user for each prompt
    for puid in data.public_user_id.unique():
        pudf = data[data.public_user_id == puid]
        uix = pudf.uix.values[0]
        # mean_ger_levels = np.mean([json.loads( s ) for s in pudf.mean_GERs_human_final.values], axis=-1)
        start_ger = np.mean(json.loads(pudf.edits_human_final.values[0]))
        start_cefr = pudf.cefr_numeric_human.values[0]

        end_ger = np.mean( json.loads( pudf.edits_human_final.values[-1] ))
        end_cefr = pudf.cefr_numeric_human.values[-1]
        prompt_level = pudf.prompt_level.values[0]
        outdata_list.append([puid, uix, start_ger, end_ger, start_cefr, end_cefr, prompt_level])

    outdata = pd.DataFrame(outdata_list, columns=["public_user_id","uix","ger_start","ger_end","cefr_start","cefr_end","prompt_level"])
    return outdata

def generate_random_colour():
    return '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))

from matplotlib import pyplot as plt
def plot_user_arrow_scatter(items):
    fig, ax = plt.subplots()
    # ax_ix = 0
    # ax = axs[ax_ix]
    # ax = axs

    user_col_dict = {"Beginner":"orange", "Intermediate":"blue", "Advanced":"green"}
    # this_key = None
    # c_list = items.uix.values
    x_starts = items.ger_start.values
    x_ends   = items.ger_end.values
    y_starts = items.cefr_start.values
    y_ends   = items.cefr_end.values
    user_cols = [user_col_dict[lev] for lev in items.prompt_level.values]
    for i in range(len(items)):
        # uix = uix_list[i]
        # pl = items.uix.values[i]
        pl = items.prompt_level.values[i]
        # print(pl)
        # if c_list[i] != this_key:
            # if i>0:
            #     plt.scatter(items[i-1, 0], items[i-1, 1], color=user_col, alpha=1, marker="x")
            # this_key = c_list[i]

        # if not (pl in user_col_dict):
        #     print(f"{pl} not in col dict, dick")
        #     user_col_dict[pl] = generate_random_colour()
        user_col = user_col_dict[pl]

            # plt.scatter(items[i, 0], items[i, 1], color=user_col, alpha=1, marker="o")
        # ax.scatter(x_ends[i], y_ends[i], color=user_col, alpha=0, marker="o")
        # else:
        x0,x1, y0,y1 = x_starts[i], x_ends[i], y_starts[i], y_ends[i]
        ax.annotate("", xytext=(x0, y0), xy=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=user_col, alpha=0.3))

        # ax.scatter(x0, y0, color=user_col, alpha=0.5, marker=".")
    ax.scatter(x_ends, y_ends, color=user_cols, alpha=1, marker=".")
    ax.scatter(x_starts, y_starts, color=user_cols, alpha=0.3, marker="x")

        # ax.scatter(x1, y1, color=user_col, alpha=0, marker=".")
    # plt.savefig("test.png")
    # plt.show()
    # fig = plt.gcf()
    return fig

uniq_user_df = get_arrow_scatter_data()
uniq_user_overall_delta_df = get_unique_user_final_delta_data()
user_component_delta_df = get_componentwise_ger_delta_data()


fig_item_scatter = uniq_user_df.hvplot.scatter(
    title="Student final mean GERs vs final CEFRs",
    x = "ger_end",
    y = "cefr_end",
    value_label= "Value",
    color="prompt_level"
)
arrow_scatter_fig = plot_user_arrow_scatter(uniq_user_df)

fig_user_final_delta_scatter = uniq_user_overall_delta_df.hvplot.scatter(
    title="Student total GER delta vs total CEFR delta (normalised)",
    x = "overall_ger_delta",
    y = "overall_cefr_delta",
    value_label= "Value",
    color="prompt_level"
)


ge_delta_bar_chart = user_component_delta_df.hvplot.bar(
    title="Delta of GE Rates for all users",
    x = "ge_type",
    y = "component_ger_delta",
    rot=90,
    ylabel="GE Type",
    xlabel="GE Rate",
    # xlim=(min_datetime, max_datetime),
    # color=ACCENT,
)

typ_final_ger_df = get_typical_final_GER()
typ_final_bar_chart = typ_final_ger_df.hvplot.bar(
    title="Final GE Rates for all users",
    x = "ge_type",
    y = "component_ger_delta",
    rot=90,
    ylabel="GE Type",
    xlabel="GE Rate",
)

# Display Data

# image = pn.pane.JPG("./imgs/roboto_teacher_chan.jpg")
image = pn.pane.JPG("""https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimg.freepik.com%2Fpremium-photo%2Fphotograph-robotic-teacher-stands-chalkboard-head-class-full-robots-stu_94574-17917.jpg&f=1&nofb=1&ipt=d37615db5633805ed54543f5bb08f32035900af398199cfb0cd4f5e278cd66f6""")

fig_item_scatter = pn.pane.HoloViews(fig_item_scatter, sizing_mode="stretch_both", name="Item Scatter")
fig_user_final_delta_scatter = pn.pane.HoloViews(fig_user_final_delta_scatter, sizing_mode="stretch_both", name="User Delta Scatter")
ge_delta_bar_chart = pn.pane.HoloViews(ge_delta_bar_chart, sizing_mode="stretch_both", name="GE Delta Bar Chart")
arrow_scatter_fig = pn.pane.Matplotlib(arrow_scatter_fig, sizing_mode="stretch_both")
typ_final_bar_chart = pn.pane.HoloViews(typ_final_bar_chart, sizing_mode="stretch_both", name="Typical Final Version GE Rates")


pn.template.FastListTemplate(
    title="WI Students Summary",
    sidebar=[image, nav],
    # main=[pn.Column(indicators, tabs, matplot,)],  #sizing_mode="stretch_both")],
    main=[ fig_item_scatter, arrow_scatter_fig, fig_user_final_delta_scatter, ge_delta_bar_chart, typ_final_bar_chart],
    # main=[ fig_item_scatter, ],
    main_layout=None,
    accent=ACCENT,
).servable()

