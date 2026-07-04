import json

import pandas
import pandas as pd
import numpy as np

def epoch_correct(source_data):
    df_list = []
    for uid in source_data.public_user_id.unique():
        this_view = source_data[source_data.public_user_id==uid].copy()
        max_ts = this_view.created_epoch.max()
        this_view["created_epoch"] = (this_view["created_epoch"] - max_ts)
        this_view["plot_time"] = this_view["created_epoch"]
        df_list.append(this_view)
    out_df = pd.concat(df_list)
    return out_df

def get_final_row_df(source_data):
    user_list = []
    for puid_ in source_data.public_user_id.unique():
        u_data = source_data[source_data.public_user_id == puid_].tail(1)
        user_list.append(u_data)
    return pd.concat(user_list)

def get_promptfinal_row_df(puid, source_data):
    user_list = []
    for puid_ in source_data.user_prompt.unique():
        u_data = source_data[source_data.user_prompt == puid_].tail(1)
        user_list.append(u_data)
    df = pd.concat(user_list)
    df = df[df.user_prompt == puid]
    return df

def get_mean_evp_components(evp_lookup, source_data):
    evps = [json.loads(e) for e in source_data.EVP_counts.values]
    univ_mean_evps = np.mean(evps, axis=0)
    print([ix for ix in evp_lookup.values()])
    v_lev = [univ_mean_evps[ix] for ix in evp_lookup.values()]
    # weighted_levels = [evp[evp_lookup[k]] * evp_lookup[k] for k in list(evp_lookup.keys())[0:-1]]
    # mean_evp = np.mean(weighted_levels)
    mean_evp_df = pd.DataFrame({"EVP_level": v_lev, "EVP_cat": evp_lookup.keys()})
    return mean_evp_df

def sigmoid(z):
    return 1 / (1+np.exp(-np.array(z)))

def get_mean_ge_components(source_data, edit_selector):
    ges = [json.loads(e) for e in source_data[edit_selector].values]
    univ_mean_ges = np.mean(ges, axis=0)
    # v_lev = [univ_mean_evps[ix] for ix in evp_lookup.values()]
    # weighted_levels = [evp[evp_lookup[k]] * evp_lookup[k] for k in list(evp_lookup.keys())[0:-1]]
    # mean_evp = np.mean(weighted_levels)
    mean_ge_df = pd.DataFrame({"GE_level": univ_mean_ges})
    return mean_ge_df



def get_final_evp_data(puid, mean_evp_components_df, source_data, evp_lookup):
    evp = source_data[source_data.public_user_id==puid].EVP_counts.tail(1).values[0]
    evp = json.loads(evp)
    v_lev = [evp[ix] for ix in evp_lookup.values()]
    weighted_levels = [evp[evp_lookup[k]] * evp_lookup[k] for k in list(evp_lookup.keys())[0:-1]]
    mean_evp = np.mean(weighted_levels)
    evp_df = pd.DataFrame({"EVP_level":v_lev, "EVP_cat":evp_lookup.keys()})
    evp_df["EVP_mean"] = mean_evp_components_df["EVP_level"]
    return evp_df, mean_evp

def get_rank_df(final_row_df, evp_lookup):
    user_list = []
    for puid_ in final_row_df.public_user_id.unique():
        u_data = final_row_df[final_row_df.public_user_id == puid_]
        evp = u_data.EVP_counts.tail(1).values[0]
        evp = json.loads(evp)
        weighted_levels = [evp[evp_lookup[k]] * evp_lookup[k] for k in list(evp_lookup.keys())[0:-1]]
        mean_evp = np.mean(weighted_levels)
        u_data["w_mean_evp"] = mean_evp

        mean_ger = np.mean(json.loads(u_data.edits_human_final.tail(1).values[0]))
        u_data["mean_ger"] = mean_ger

        mean_c = u_data.cefr_numeric_human.tail(1).values[0]
        u_data["cefr"] = mean_c

        user_list.append(u_data)
    final_rows_data = pd.concat(user_list)
    final_rows_data["mean_evp_rank"] = final_rows_data["w_mean_evp"].rank(ascending=False)
    final_rows_data["mean_ger_rank"] = final_rows_data["mean_ger"].rank()
    final_rows_data["mean_cefr_rank"] = final_rows_data["cefr"].rank(ascending=False)
    num_users = len(final_rows_data)
    return final_rows_data, num_users

def get_rankings(rank_df, public_user_id):
    user_row  = rank_df[rank_df.public_user_id==public_user_id]
    mean_ger = user_row.mean_ger_rank.values[0]
    mean_cefr = user_row.mean_cefr_rank.values[0]
    mean_evp = user_row.mean_evp_rank.values[0]
    # mean_ger = float(user_row.mean_ger_rank.values)
    # mean_cefr = float(user_row.mean_cefr_rank.values)
    # mean_evp = float(user_row.mean_evp_rank.values)
    ranked_out_of = len(rank_df)
    return mean_ger, mean_cefr, mean_evp, ranked_out_of

def ger_filter_data(source_data, edit_selector, public_user_id):
    data = source_data[(source_data.public_user_id == public_user_id)] #& (source_data.created_epoch <= epoch_ts)]
    data.sort_values(by=["created_epoch"], inplace=True)
    # ger_level = [np.mean(json.loads(v)) for v in data["edits_human_final"].values]
    # data["mean_ger_level_human"] = ger_level

    ger_level = [np.mean(json.loads(v)) for v in data.edits_human_final.values]
    data["mean_ger_level_human"] = ger_level
    # ger_level = [np.mean(json.loads(v)) for v in data.edits_human_all.values]
    # data["mean_ger_level_all"] = ger_level
    ger_level = [np.mean(json.loads(v)) for v in data.edits_this_LT.values]
    data["mean_ger_edits_LT"] = ger_level
    pred_ger_level = [np.mean(json.loads(v)[0:-1]) for v in data.predicted_GER.values]
    data["ai_pred_ger_level"] = pred_ger_level

    return data[["created_epoch", "cefr_numeric", "cefr_numeric_human", "mean_ger_level_human", "mean_ger_edits_LT", "ai_pred_ger_level", "predicted_CEFR", "plot_time"]]

def get_promptfinal_data(source_data:pandas.DataFrame, edit_selector:str) -> pd.DataFrame:
    user_list = []
    for puid_ in source_data.user_prompt.unique():
        u_data = source_data[source_data.user_prompt == puid_].tail(1)
        user_list.append(u_data)
    df = pd.concat(user_list)
    data = df #[(df.public_user_id == public_user_id)] ##& (df.created_epoch <= epoch_ts)]
    # data.sort_values(by=["created_epoch"], inplace=True)
    ger_level = [np.mean(json.loads(v)) for v in data[edit_selector].values]
    data["mean_ger_level_human"] = ger_level
    # ger_level = [np.mean(json.loads(v)) for v in data.edits_human_all.values]
    # data["mean_ger_level_all"] = ger_level
    ger_level = [np.mean(json.loads(v)) for v in data.edits_this_LT.values]
    data["mean_ger_edits_LT"] = ger_level
    pred_ger_level = [np.mean(json.loads(v)) for v in data.predicted_GER.values]
    data["ai_pred_ger_level"] = pred_ger_level

    return data #[["created_epoch", "mean_ger_level_hfinal", "mean_ger_edits_LT", "cefr_numeric", "cefr_numeric_human", "ai_pred_ger_level", "predicted_CEFR"]]


def get_final_data_from_prompt_final(pf_df):
    user_list = []
    for puid_ in pf_df.public_user_id.unique():
        u_data = pf_df[pf_df.public_user_id == puid_].tail(1)
        user_list.append(u_data)
    out_df = pd.concat(user_list)
    return out_df


def get_this_user_promptfinal(public_user_id, promptfinal_df):
    this_user_df = promptfinal_df[promptfinal_df.public_user_id==public_user_id]
    return this_user_df


def calc_ger_delta(ger_df):
    gers = ger_df.ai_pred_ger_level
    L = len(gers)
    st_g = gers.head(1).values[0]
    ed_g = gers.tail(1).values[0]
    delta = ed_g - st_g
    # delta = ed_g - st_g
    rel_delta = 10000*(delta / L)
    return delta, rel_delta


def calc_cefr_delta(ger_df):
    cefrs = ger_df.cefr_numeric_human
    L = len(cefrs)
    st_c = cefrs.head(1).values[0]
    ed_c = cefrs.tail(1).values[0]
    delta = ed_c - st_c
    # delta = float(ed_c - st_c)
    rel_delta = (delta / L)
    return delta, rel_delta
