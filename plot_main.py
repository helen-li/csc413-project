# Compute Plots for different purposes

import seaborn as sns
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

EPS = 1e-8

# Set Plotting Variables

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")#, font_scale=1.1)
font = {'family' : 'Open Sans',
        'size'   : 24}

mpl.rc('font', **font)
colors = ['#31356e', '#704e85', '#a66c98', '#d690aa']
explode=[0.05]*4

model_names = ['GT-Modular', 'Modular-op', 'Modular', 'Monolithic']

def plot_pie(arr, labels, colors, name, circle=True):
    def filter(arr, lab, color):
        a = []
        l = []
        c = []
        for i in range(len(arr)):
            if arr[i] == 0:
                continue
            a.append(arr[i])
            l.append(lab[i])
            c.append(color[i])

        return a, l, c

    arr, lab, col = filter(arr, labels, colors)
    _, texts, autotexts = plt.pie(arr,
                                  colors=col,
                                  labels=lab,
                                  autopct='%.1f%%',
                                  startangle=90,
                                  pctdistance=0.4
                                  )

    for text in texts:
        text.set_color('#404040')
        if not circle:
            text.set_color('#000000')
        text.set_fontsize(22)
    for autotext in autotexts:
        autotext.set_color('#404040')
        if not circle:
            autotext.set_color('#000000')
        autotext.set_fontsize(22)

    # draw circle
    if circle:
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.gca().axis('equal')
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_line(df, x, y, col, row, hue, name, ci=None, hue_order = model_names, title=None, xlabel = None, ylabel = None, marker=None):
    g = sns.relplot(
        data=df, x=x, y=y, col=col, row=row,
        hue=hue, kind="line", facet_kws={'sharey': False},
        hue_order=hue_order, ci=ci, marker=marker
    )
    (g.set_titles(title)
     .set_ylabels(ylabel, clear_inner=False)
     .set_xlabels(xlabel, clear_inner=False)
     .despine(left=True))

    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.02),
        ncol=4,
        title=None,
        frameon=False,
        fontsize=11,
    )

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def add_random(df, mode, arch='MLP'):
    rules = [2, 4, 8, 16, 32]
    encs = [32, 64, 128, 256, 512]
    dims = [128, 256, 512, 1024, 2048]
    modes = ['last', 'best']

    def collapse_metric_worse(prob, rules):
        p = np.min(np.sum(prob, axis=0))
        cmw = 1 - rules * p
        return cmw

    def collapse_metric(prob, rules):
        p = np.sum(prob, axis=0)
        cm = rules * np.sum(np.maximum(np.ones_like(p) / rules - p, 0)) / (rules - 1)
        return cm

    def mutual_info(prob):
        m1 = np.sum(prob, axis=0, keepdims=True)
        m2 = np.sum(prob, axis=1, keepdims=True)
        m = m1 * m2
        return np.sum(prob * np.log(prob / (m + EPS) + EPS))

    def specialization_score(true_p, empirical_p):
        true_p.sort()
        empirical_p.sort()

        return np.sum(np.abs(true_p - empirical_p)) / 2.

    def specialization_metric(prob, rules):
        spec = 0.
        p_ = np.sum(prob, axis=0)

        for eval_seed in range(100):
            rng = np.random.RandomState(eval_seed)
            p = rng.dirichlet(alpha = np.ones(rules))
            spec += specialization_score(p, p_)

        return spec / 100.

    def hungarian_metric(prob, rules):
        prob = prob * rules
        cost = 1 - prob
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros((rules, rules))
        perm[row_ind, col_ind] = 1.
        hung_score = np.sum(np.abs(perm - prob)) / (2 * rules)
        return hung_score

    for rule in [2, 4, 8, 16, 32]:
        prob = np.ones((rule, rule)) / (rule * rule)
        cmw = collapse_metric_worse(prob, rule)
        cm = collapse_metric(prob, rule)
        mi = mutual_info(prob)
        spec = specialization_metric(prob, rule)
        hung = hungarian_metric(prob, rule)

        for enc, dim in zip(encs, dims):
            for m in modes:
                if arch == 'MLP':
                    df.loc[-1] = [mode, 'Random', rule, enc, dim, -1, -1, -1, -1, -1,
                                  cm, cmw, spec, mi, hung, m]
                elif arch == 'MHA':
                    df.loc[-1] = [mode, 'Random', rule, 1, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                    df.index += 1
                    df.loc[-1] = [mode, 'Random', rule, 2, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                elif arch == 'RNN':
                    df.loc[-1] = [mode, 'Random', rule, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                df.index += 1

folder = ['Logistics_0.0', 'Logistics_0.1', 'Logistics_0.25', 'Logistics_0.5', 'Logistics_1.0', 'Logistics_2.0']
noises = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
# Get MLP Details

print('MLP')

mlp_versions = ['MLP-Classification-WithDecoder', 'MLP-Classification-WithoutDecoder', 'MLP-Regression-WithDecoder', 'MLP-Regression-WithoutDecoder']
names = [f'MLP/Classification/With_Decoder', f'MLP/Classification/Without_Decoder', f'MLP/Regression/With_Decoder', f'MLP/Regression/Without_Decoder']

mlp_clf_wd_final = None
mlp_reg_wd_final = None

mlp_clf_wd_training = None
mlp_reg_wd_training = None

for i in range(6):
    mlp_clf_wd = pd.read_pickle(f'MLP/Classification/{folder[i]}/df_final.pt')
    mlp_clf_wd['Perf'] = 100. - mlp_clf_wd['Perf']
    mlp_clf_wd['Perf-OoD'] = 100. - mlp_clf_wd['Perf-OoD']
    add_random(mlp_clf_wd, mlp_versions[0])
    mlp_clf_wd['Mutual Information'] = (np.log(mlp_clf_wd['Rules'].astype(float)) - mlp_clf_wd['Mutual Information']) / np.log(mlp_clf_wd['Rules'].astype(float))
    mlp_clf_wd['Noise'] = noises[i]
    if mlp_clf_wd_final is None:
        mlp_clf_wd_final = mlp_clf_wd
    else:
        mlp_clf_wd_final = pd.concat([mlp_clf_wd_final, mlp_clf_wd], ignore_index=True)

    mlp_clf_wd_t = pd.read_pickle(f'MLP/Classification/{folder[i]}/df_training.pt')
    mlp_clf_wd_t['Perf'] = 100. - mlp_clf_wd_t['Perf']
    mlp_clf_wd_t['Noise'] = noises[i]
    if mlp_clf_wd_training is None:
        mlp_clf_wd_training = mlp_clf_wd_t
    else:
        mlp_clf_wd_training = pd.concat([mlp_clf_wd_training, mlp_clf_wd_t], ignore_index=True)

    mlp_reg_wd = pd.read_pickle(f'MLP/Regression/{folder[i]}/df_final.pt')
    add_random(mlp_reg_wd, mlp_versions[2])
    mlp_reg_wd['Mutual Information'] = (np.log(mlp_reg_wd['Rules'].astype(float)) - mlp_reg_wd['Mutual Information']) / np.log(mlp_reg_wd['Rules'].astype(float))
    mlp_reg_wd['Noise'] = noises[i]
    if mlp_reg_wd_final is None:
        mlp_reg_wd_final = mlp_reg_wd
    else:
        mlp_reg_wd_final = pd.concat([mlp_reg_wd_final, mlp_reg_wd], ignore_index=True)
    
    mlp_reg_wd_t = pd.read_pickle(f'MLP/Regression/{folder[i]}/df_training.pt')
    mlp_reg_wd_t['Noise'] = noises[i]
    if mlp_reg_wd_training is None:
        mlp_reg_wd_training = mlp_reg_wd_t
    else:
        mlp_reg_wd_training = pd.concat([mlp_reg_wd_training, mlp_reg_wd_t], ignore_index=True)

with open(f'MLP/Classification/Logistics_0.0/ranking.pickle', 'rb') as handle:
    mlp_clf_wd_rank = pickle.load(handle)
mlp_clf_wd_final['Arch'] = 'Classification'

with open(f'MLP/Regression/Logistics_0.0/ranking.pickle', 'rb') as handle:
    mlp_reg_wd_rank = pickle.load(handle)
mlp_reg_wd_final['Arch'] = 'Regression'

mlp_final = pd.concat([mlp_clf_wd_final, mlp_reg_wd_final], ignore_index=True)
mlp_rank = dict()
for key in mlp_clf_wd_rank.keys():
    mlp_rank[key] = mlp_clf_wd_rank[key] + mlp_reg_wd_rank[key]
    for i in range(1, 6):
        with open(f'MLP/Classification/Logistics_{noises[i]}/ranking.pickle', 'rb') as handle:
            mlp_rank[key] += pickle.load(handle)[key]
        with open(f'MLP/Regression/Logistics_{noises[i]}/ranking.pickle', 'rb') as handle:
            mlp_rank[key] += pickle.load(handle)[key]

mlp_training = pd.concat([mlp_clf_wd_training, mlp_reg_wd_training], ignore_index=True)

# Get Transformer Details

print('MHA')

tsf_versions = ['MHA-Classification', 'MHA-Regression']
names = [f'MHA/Classification', f'MHA/Regression']
lens = [3, 5, 10]

mha_clf_final = None
mha_reg_final = None

mha_clf_training = None
mha_reg_training = None

for i in range(6):
    mha_clf = pd.read_pickle(f'MHA/Classification/{folder[i]}/df_final.pt')
    for l in lens:
        mha_clf[f'Perf-{l}'] = 100. - mha_clf[f'Perf-{l}']
        mha_clf[f'Perf-OoD-{l}'] = 100. - mha_clf[f'Perf-OoD-{l}']
    add_random(mha_clf, tsf_versions[0], arch='MHA')
    mha_clf['Mutual Information'] = (np.log(mha_clf['Rules'].astype(float)) - mha_clf['Mutual Information']) / np.log(mha_clf['Rules'].astype(float))
    mha_clf['Perf'] = mha_clf['Perf-10']
    mha_clf['Perf-OoD'] = mha_clf['Perf-OoD-30']
    mha_clf['Noise'] = noises[i]
    if mha_clf_final is None:
        mha_clf_final = mha_clf
    else:
        mha_clf_final = pd.concat([mha_clf_final, mha_clf], ignore_index=True)
    
    mha_clf_t = pd.read_pickle(f'MHA/Classification/{folder[i]}/df_training.pt')
    mha_clf_t['Perf'] = 100. - mha_clf_t['Perf']
    mha_clf_t['Noise'] = noises[i]
    if mha_clf_training is None:
        mha_clf_training = mha_clf_t
    else:
        mha_clf_training = pd.concat([mha_clf_training, mha_clf_t], ignore_index=True)
    
    mha_reg = pd.read_pickle(f'MHA/Regression/{folder[i]}/df_final.pt')
    add_random(mha_reg, tsf_versions[0], arch='MHA')
    mha_reg['Mutual Information'] = (np.log(mha_reg['Rules'].astype(float)) - mha_reg['Mutual Information']) / np.log(mha_reg['Rules'].astype(float))
    mha_reg['Perf'] = mha_reg['Perf-10']
    mha_reg['Perf-OoD'] = mha_reg['Perf-OoD-30']
    mha_reg['Noise'] = noises[i]
    if mha_reg_final is None:
        mha_reg_final = mha_reg
    else:
        mha_reg_final = pd.concat([mha_reg_final, mha_reg], ignore_index=True)

    mha_reg_t = pd.read_pickle(f'MHA/Regression/{folder[i]}/df_training.pt')
    mha_reg_t['Noise'] = noises[i]
    if mha_reg_training is None:
        mha_reg_training = mha_reg_t
    else:
        mha_reg_training = pd.concat([mha_reg_training, mha_reg_t], ignore_index=True)

mha_clf_final['Arch'] = 'Classification'
with open(f'MHA/Classification/Logistics_0.0/ranking_1.pickle', 'rb') as handle:
    mha_clf_rank_1 = pickle.load(handle)

mha_reg_final['Arch'] = 'Regression'
with open(f'MHA/Regression/Logistics_0.0/ranking_1.pickle', 'rb') as handle:
    mha_reg_rank_1 = pickle.load(handle)

mha_final = pd.concat([mha_clf_final, mha_reg_final], ignore_index=True)
mha_training = pd.concat([mha_clf_training, mha_reg_training], ignore_index=True)
mha_rank_1 = dict()
for key in mha_clf_rank_1.keys():
    mha_rank_1[key] = mha_clf_rank_1[key] + mha_reg_rank_1[key]
    for i in range(1, 6):
        with open(f'MHA/Classification/Logistics_{noises[i]}/ranking_1.pickle', 'rb') as handle:
            mha_rank_1[key] += pickle.load(handle)[key]
        with open(f'MHA/Regression/Logistics_{noises[i]}/ranking_1.pickle', 'rb') as handle:
            mha_rank_1[key] += pickle.load(handle)[key]

# Get RNN Details

print('RNN')

rnn_versions = ['RNN-Classification', 'RNN-Regression']
names = [f'RNN/SCOFF/Classification', f'RNN/SCOFF/Regression']
lens = [3, 5, 10]

rnn_clf_final = None
rnn_reg_final = None
rnn_clf_training = None
rnn_reg_training = None

for i in range(6):
    rnn_clf = pd.read_pickle(f'RNN/Classification/{folder[i]}/df_final.pt')
    for l in lens:
        rnn_clf[f'Perf-{l}'] = 100. - rnn_clf[f'Perf-{l}']
        rnn_clf[f'Perf-OoD-{l}'] = 100. - rnn_clf[f'Perf-OoD-{l}']
    add_random(rnn_clf, rnn_versions[0], arch='RNN')
    rnn_clf['Mutual Information'] = (np.log(rnn_clf['Rules'].astype(float)) - rnn_clf['Mutual Information']) / np.log(rnn_clf['Rules'].astype(float))
    rnn_clf['Perf'] = rnn_clf['Perf-10']
    rnn_clf['Perf-OoD'] = rnn_clf['Perf-OoD-30']
    rnn_clf['Noise'] = noises[i]
    if rnn_clf_final is None:
        rnn_clf_final = rnn_clf
    else:
        rnn_clf_final = pd.concat([rnn_clf_final, rnn_clf], ignore_index=True)
    
    rnn_clf_t = pd.read_pickle(f'RNN/Classification/{folder[i]}/df_training.pt')
    rnn_clf_t['Perf'] = 100. - rnn_clf_t['Perf']
    rnn_clf_t['Noise'] = noises[i]
    if rnn_clf_training is None:
        rnn_clf_training = rnn_clf_t
    else:
        rnn_clf_training = pd.concat([rnn_clf_training, rnn_clf_t], ignore_index=True)

    rnn_reg = pd.read_pickle(f'RNN/Regression/{folder[i]}/df_final.pt')
    add_random(rnn_reg, rnn_versions[0], arch='RNN')
    rnn_reg['Mutual Information'] = (np.log(rnn_reg['Rules'].astype(float)) - rnn_reg['Mutual Information']) / np.log(rnn_reg['Rules'].astype(float))
    rnn_reg['Perf'] = rnn_reg['Perf-10']
    rnn_reg['Perf-OoD'] = rnn_reg['Perf-OoD-30']
    rnn_reg['Noise'] = noises[i]
    if rnn_reg_final is None:
        rnn_reg_final = rnn_reg
    else:
        rnn_reg_final = pd.concat([rnn_reg_final, rnn_reg], ignore_index=True)
    
    rnn_reg_t = pd.read_pickle(f'RNN/Regression/{folder[i]}/df_training.pt')
    rnn_reg_t['Noise'] = noises[i]
    if rnn_reg_training is None:
        rnn_reg_training = rnn_reg_t
    else:
        rnn_reg_training = pd.concat([rnn_reg_training, rnn_reg_t], ignore_index=True)

rnn_clf_final['Arch'] = 'Classification'
with open(f'RNN/Classification/Logistics_0.0/ranking.pickle', 'rb') as handle:
    rnn_clf_rank = pickle.load(handle)

rnn_reg_final['Arch'] = 'Regression'
with open(f'RNN/Regression/Logistics_0.0/ranking.pickle', 'rb') as handle:
    rnn_reg_rank = pickle.load(handle)

rnn_final = pd.concat([rnn_clf_final, rnn_reg_final], ignore_index=True)
rnn_training = pd.concat([rnn_clf_training, rnn_reg_training], ignore_index=True)
rnn_rank = dict()
for key in rnn_clf_rank.keys():
    rnn_rank[key] = rnn_clf_rank[key] + rnn_reg_rank[key]
    for i in range(1, 6):
        with open(f'RNN/Classification/Logistics_{noises[i]}/ranking.pickle', 'rb') as handle:
            rnn_rank[key] += pickle.load(handle)[key]
        with open(f'RNN/Regression/Logistics_{noises[i]}/ranking.pickle', 'rb') as handle:
            rnn_rank[key] += pickle.load(handle)[key]

# Get Consolidated Details

print('Consolidated')

rank = dict()
for key in mha_rank_1.keys():
    rank[key] = mha_rank_1[key] + mlp_rank[key] + rnn_rank[key]

hue_pie = ['GT-Modular', 'Modular', 'Modular-op', 'Monolithic']
hue_pie_sub = ['Modular', 'Monolithic']

df_final = pd.concat([mlp_final, mha_final, rnn_final], ignore_index=True)
df_training = pd.concat([mlp_training, mha_training, rnn_training], ignore_index=True)

df_final = df_final.drop(['Perf-3', 'Perf-5', 'Perf-10', 'Perf-20', 'Perf-30',
                         'Perf-OoD-3', 'Perf-OoD-5', 'Perf-OoD-10', 'Perf-OoD-20', 'Perf-OoD-30'],
                         axis = 1)
df_final = df_final.rename(columns=
                           {
                               'Mutual Information': 'Inverse Mutual Information',
                               'Specialization': 'Adaptation',
                               'Collapse': 'Collapse-Avg',
                               'Collapse Worst': 'Collapse-Worst',
                               'Perf': 'ID',
                               # 'Perf': 'In-Distribution',
                               'Perf-OoD': 'OoD',
                               # 'Perf-OoD': 'Out-of-Distribution',
                               'Hungarian Score': 'Alignment'
                           })
df_final = df_final.melt(id_vars=['Arch', 'Mode', 'Model', 'Rules', 'Encoder Dimension', 'Dimension', 'Data Seed', 'Seed', 'Number of Parameters', 'Eval Mode', 'Search-Version', 'Noise'],
                         value_vars=['ID', 'OoD', 'Collapse-Avg', 'Collapse-Worst', 'Adaptation', 'Inverse Mutual Information', 'Alignment'],
                         # value_vars=['In-Distribution', 'Out-of-Distribution', 'Collapse-Avg', 'Collapse-Worst', 'Adaptation', 'Inverse Mutual Information', 'Hungarian'],
                         var_name='Metric', value_name='Score')

d_collapse = df_final[df_final['Metric'].isin(['Collapse-Avg', 'Collapse-Worst'])]
d_collapse = d_collapse[d_collapse['Model'] != 'Monolithic']

d_spec = df_final[df_final['Metric'].isin(['Inverse Mutual Information', 'Adaptation'])]
d_spec = d_spec[d_spec['Model'] != 'Monolithic']

d_metrics = df_final[df_final['Metric'].isin(['Collapse-Avg', 'Collapse-Worst', 'Inverse Mutual Information', 'Adaptation', 'Alignment'])]
d_metrics = d_metrics[d_metrics['Model'] != 'Monolithic']

# d_perf = df_final[df_final['Metric'].isin(['In-Distribution', 'Out-of-Distribution'])]
d_perf = df_final[df_final['Metric'].isin(['ID', 'OoD'])]
d_perf['Metric'] = d_perf['Arch'] + ' | ' + d_perf['Metric']
d_perf = d_perf[d_perf['Model'] != 'Random']

d_perf_mlp = d_perf[d_perf['Mode'].isin(['MLP-Classification-WithDecoder', 'MLP-Regression-WithDecoder'])]
d_perf_mha = d_perf[d_perf['Mode'].isin(['MHA-Classification', 'MHA-Regression'])]
d_perf_rnn = d_perf[d_perf['Mode'].isin(['SCOFF-Classification', 'SCOFF-Regression'])]
hue_perf = ['GT-Modular', 'Modular-op', 'Modular', 'Monolithic']

hue_metrics = ['GT-Modular', 'Modular-op', 'Modular', 'Random']

g = sns.catplot(data=d_collapse, x='Rules', y='Score', hue='Model',
                hue_order=hue_metrics, kind='bar', col='Metric',
                alpha=0.9, saturation=0.8, sharey=False)

(g.set_axis_labels("Rules", "Metric Score", fontsize=20)
  .set_titles("{col_name}", size=20)
  .despine(left=True))
g._legend.set_title("")
plt.setp(g._legend.get_texts(), fontsize=16)

g.set_xticklabels(size = 16)
g.set_yticklabels(size = 16)
g._legend.set(bbox_to_anchor=(1.01, 0.5))

plt.savefig('Main_Plots/collapse_r.pdf', bbox_inches='tight')
plt.close()

g = sns.catplot(data=d_collapse, x='Model', y='Score', hue='Model',
                hue_order=hue_metrics, kind='bar', col='Metric',
                alpha=0.9, saturation=0.8, dodge=False, sharey=False)

(g.set_axis_labels("", "Metric Score", fontsize=20)
  .set_xticklabels([])
  .set_titles("{col_name}", size=20)
  .despine(left=True))
g.set_yticklabels(size = 16)

g.axes[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor="white", edgecolor='white', fontsize=20)

plt.savefig('Main_Plots/collapse.pdf', bbox_inches='tight')
plt.close()


g = sns.catplot(data=d_spec, x='Rules', y='Score', hue='Model',
                hue_order=hue_metrics, kind='bar', col='Metric',
                alpha=0.9, saturation=0.8, sharey=False)

(g.set_axis_labels("Rules", "Metric Score", fontsize=20)
  .set_titles("{col_name}", size=20)
  .despine(left=True))
g.set_xticklabels(size = 16)
g.set_yticklabels(size = 16)

g._legend.set_title("")
plt.setp(g._legend.get_texts(), fontsize=20)
g._legend.set(bbox_to_anchor=(1.01, 0.5))

plt.savefig('Main_Plots/spec_r.pdf', bbox_inches='tight')
plt.close()

g = sns.catplot(data=d_spec, x='Model', y='Score', hue='Model',
                hue_order=hue_metrics, kind='bar', col='Metric',
                alpha=0.9, saturation=0.8, dodge=False, sharey=False)

(g.set_axis_labels("", "Metric Score", fontsize=20)
  .set_xticklabels([])
  .set_titles("{col_name}", size=20)
  .despine(left=True))
g.set_yticklabels(size = 16)

g.axes[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor="white", edgecolor='white', fontsize=20)

plt.savefig('Main_Plots/spec.pdf', bbox_inches='tight')
plt.close()

d_metrics_collapse_avg = d_metrics[d_metrics['Metric'] == 'Collapse-Avg']
d_metrics_collapse_worst = d_metrics[d_metrics['Metric'] == 'Collapse-Worst']
d_metrics_alignment = d_metrics[d_metrics['Metric'] == 'Alignment']
d_metrics_adaptation = d_metrics[d_metrics['Metric'] == 'Adaptation']
d_metrics_mi = d_metrics[d_metrics['Metric'] == 'Inverse Mutual Information']

d_metrics_list = [d_metrics_collapse_avg, d_metrics_collapse_worst, d_metrics_alignment, d_metrics_adaptation, d_metrics_mi]
d_metrics_names = ['Collapse-Avg', 'Collapse-Worst', 'Alignment', 'Adaptation', 'Inverse Mutual Information']
d_metric_y = [1.0, 1.1, 1.0, 0.5, 1.1]
d_modes = [
    ['MLP-Classification-WithDecoder', 'MLP-Regression-WithDecoder'],
    ['MHA-Classification', 'MHA-Regression'],
    ['SCOFF-Classification', 'SCOFF-Regression']
]

for m_index in range(5):
    for mode_index in range(3): 
        d_filtered = d_metrics_list[m_index][d_metrics_list[m_index]['Mode'].isin(d_modes[mode_index])]
        g = sns.catplot(data=d_filtered, x='Noise', y='Score', hue='Model',
                        hue_order=hue_metrics, kind='bar', col='Rules',
                        alpha=0.9, saturation=0.8, sharey=True,
                        col_order=[2, 4, 8, 16, 32])

        (g.set_axis_labels("Noise", d_metrics_names[m_index], fontsize=22)
        .set_titles("{col_name} Rules", size=24)
        .despine(left=True))
        g.set_xticklabels(size = 22)

        for i,ax in enumerate(g.axes[0]):
            if i != 2:
                ax.set_xlabel('', fontsize=22)

        sns.move_legend(
            g,
            loc=(0.25, -0.01),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=24,
        )

        g.axes[0][0].set_ylim(0, d_metric_y[m_index])
        g.axes[0][0].tick_params(axis="y", labelsize=22)

        plt.savefig(f'Main_Plots/{d_metrics_names[m_index]}-{mode_index}.pdf', bbox_extra_artists=(g._legend,), bbox_inches='tight', pad_inches=1.0)
        print(f'Saved {d_metrics_names[m_index]}')
        plt.close()

g = sns.catplot(data=d_metrics, x='Model', y='Score', hue='Model',
                hue_order=hue_metrics, kind='bar', col='Metric',
                alpha=0.9, saturation=0.8, dodge=False, sharey=False,
                col_order=['Collapse-Avg', 'Collapse-Worst', 'Alignment', 'Adaptation', 'Inverse Mutual Information'])

(g.set_axis_labels("", "Metric Score", fontsize=22)
  .set_xticklabels([])
  .set_titles("{col_name}", size=24)
  .despine(left=True))
g.set_yticklabels(size = 18)
for ax in g.axes[0]:
    for index, label in enumerate(ax.get_yticklabels()):
       if index % 2 == 0:
          label.set_visible(False)
       else:
          label.set_visible(True)

g.axes[0][2].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3), facecolor="white", edgecolor='white', fontsize=24)
# sns.move_legend(
#     g, "lower center",
#     bbox_to_anchor=(.5, -0.05),
#     ncol=4,
#     title=None,
#     frameon=False,
#     fontsize=20,
# )

plt.savefig('Main_Plots/metrics.pdf', bbox_inches='tight')
plt.close()


g = sns.relplot(
    data=d_perf_mlp, x="Rules", y="Score", col='Metric',
    hue='Model', kind="line", facet_kws={'sharey': False},
    hue_order=hue_perf, ci=True, marker='o',
    col_order = ['Classification | ID', 'Classification | OoD', 'Regression | ID', 'Regression | OoD']
    # col_order = ['Classification | In-Distribution', 'Classification | Out-of-Distribution', 'Regression | In-Distribution', 'Regression | Out-of-Distribution']
)
(g.set_titles("{col_name}", size=20, style='oblique', family='monospace')
 .despine(left=True))

g.axes[0][0].set_ylabel('Error', size=20)
g.axes[0][2].set_ylabel('Loss', size=20)
g.set_xlabels('Rules', size=20)
b1 = g.axes[0][2].get_position()
b2 = g.axes[0][3].get_position()
b1.x0 = b1.x0 + 0.015
b2.x0 = b2.x0 + 0.015
b1.x1 = b1.x1 + 0.015
b2.x1 = b2.x1 + 0.015
g.axes[0][2].set_position(b1)
g.axes[0][3].set_position(b2)
g.set_xticklabels(size = 16)
g.set_yticklabels(size = 16)

g._legend.set(bbox_to_anchor=(1.018, 0.5))
g._legend.set_title("")
plt.setp(g._legend.get_texts(), fontsize=20)

sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, -0.05),
    ncol=4,
    title=None,
    frameon=False,
    fontsize=20,
)

for ax in g.axes[0]:
    for index, label in enumerate(ax.get_yticklabels()):
       if index % 2 == 0:
          label.set_visible(False)
       else:
          label.set_visible(True)

plt.savefig('Main_Plots/mlp.pdf', bbox_inches='tight')
plt.close()

g = sns.relplot(
    data=d_perf_mha, x="Rules", y="Score", col='Metric',
    hue='Model', kind="line", facet_kws={'sharey': False},
    hue_order=hue_perf, ci=True, marker='o',
    col_order = ['Classification | ID', 'Classification | OoD', 'Regression | ID', 'Regression | OoD']
    # col_order = ['Classification | In-Distribution', 'Classification | Out-of-Distribution', 'Regression | In-Distribution', 'Regression | Out-of-Distribution']
)
(g.set_titles("{col_name}", size=20, style='oblique', family='monospace')
 .despine(left=True))

g.axes[0][0].set_ylabel('Error', size=20)
g.axes[0][2].set_ylabel('Loss', size=20)
g.set_xlabels('Rules', size=20)
b1 = g.axes[0][2].get_position()
b2 = g.axes[0][3].get_position()
b1.x0 = b1.x0 + 0.01
b2.x0 = b2.x0 + 0.01
b1.x1 = b1.x1 + 0.01
b2.x1 = b2.x1 + 0.01
g.axes[0][2].set_position(b1)
g.axes[0][3].set_position(b2)
g.set_xticklabels(size = 16)
g.set_yticklabels(size = 16)

g._legend.set(bbox_to_anchor=(1.018, 0.5))
g._legend.set_title("")
plt.setp(g._legend.get_texts(), fontsize=20)

sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, -0.05),
    ncol=4,
    title=None,
    frameon=False,
    fontsize=20,
)

for ax in g.axes[0]:
    for index, label in enumerate(ax.get_yticklabels()):
       if index % 2 == 0:
          label.set_visible(False)
       else:
          label.set_visible(True)

plt.savefig('Main_Plots/mha.pdf', bbox_inches='tight')
plt.close()


g = sns.relplot(
    data=d_perf_rnn, x="Rules", y="Score", col='Metric',
    hue='Model', kind="line", facet_kws={'sharey': False},
    hue_order=hue_perf, ci=True, marker='o',
    col_order = ['Classification | ID', 'Classification | OoD', 'Regression | ID', 'Regression | OoD']
    # col_order = ['Classification | In-Distribution', 'Classification | Out-of-Distribution', 'Regression | In-Distribution', 'Regression | Out-of-Distribution']
)
(g.set_titles("{col_name}", size=20, style='oblique', family='monospace')
 .despine(left=True))

g.axes[0][0].set_ylabel('Error', size=20)
g.axes[0][2].set_ylabel('Loss', size=20)
g.set_xlabels('Rules', size=20)
b1 = g.axes[0][2].get_position()
b2 = g.axes[0][3].get_position()
b1.x0 = b1.x0 + 0.01
b2.x0 = b2.x0 + 0.01
b1.x1 = b1.x1 + 0.01
b2.x1 = b2.x1 + 0.01
g.axes[0][2].set_position(b1)
g.axes[0][3].set_position(b2)
g.set_xticklabels(size = 16)
g.set_yticklabels(size = 16)

g._legend.set(bbox_to_anchor=(1.018, 0.5))
g._legend.set_title("")
plt.setp(g._legend.get_texts(), fontsize=20)

sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, -0.05),
    ncol=4,
    title=None,
    frameon=False,
    fontsize=20,
)

for ax in g.axes[0]:
    for index, label in enumerate(ax.get_yticklabels()):
       if index % 2 == 0:
          label.set_visible(False)
       else:
          label.set_visible(True)

plt.savefig('Main_Plots/rnn.pdf', bbox_inches='tight')
plt.close()

d_perf_mlp['Architecture'] = 'MLP'
d_perf_mha['Architecture'] = 'MHA'
d_perf_rnn['Architecture'] = 'RNN'

d_perf = pd.concat([d_perf_mlp, d_perf_mha, d_perf_rnn])

rules = [2, 4, 8, 16, 32]
for r_index in range(5):
    d_perf_r = d_perf[d_perf['Rules'] == rules[r_index]]

    g = sns.relplot(
        data=d_perf_r, x="Noise", y="Score", col='Metric', row='Architecture',
        hue='Model', kind="line", facet_kws={'sharey': False},
        hue_order=hue_perf, ci=True, marker='o',
        col_order = ['Classification | ID', 'Classification | OoD', 'Regression | ID', 'Regression | OoD']
    )
    (g.set_titles("{col_name}", size=20, style='oblique', family='monospace')
    .despine(left=True))

    sns.move_legend(
        g,
        borderaxespad=-0.4,
        ncol=4,
        title=None,
        frameon=False,
        fontsize=20,
        loc=(0.3, 0.01),
    )

    g.axes[1][0].set_title('')
    g.axes[1][1].set_title('')
    g.axes[1][2].set_title('')
    g.axes[1][3].set_title('')
    g.axes[2][0].set_title('')
    g.axes[2][1].set_title('')
    g.axes[2][2].set_title('')
    g.axes[2][3].set_title('')

    for i in range(3):
        g.axes[i][0].set_ylabel('Error', size=20)
        g.axes[i][2].set_ylabel('Loss', size=20)
        g.set_xlabels('Noise', size=20)
        b1 = g.axes[i][2].get_position()
        b2 = g.axes[i][3].get_position()
        b1.x0 = b1.x0 + 0.02
        b2.x0 = b2.x0 + 0.02
        b1.x1 = b1.x1 + 0.02
        b2.x1 = b2.x1 + 0.02
        g.axes[i][2].set_position(b1)
        g.axes[i][3].set_position(b2)

    for i in range(4):
        b1 = g.axes[1][i].get_position()
        b2 = g.axes[2][i].get_position()
        b1.y0 = b1.y0 + .02
        b2.y0 = b2.y0 + .04
        b1.y1 = b1.y1 + .02
        b2.y1 = b2.y1 + .04
        g.axes[1][i].set_position(b1)
        g.axes[2][i].set_position(b2)

    g.set_xticklabels(size = 16)
    g.set_yticklabels(size = 16)

    for ax in g.axes.flatten():
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(False)
            else:
                label.set_visible(True)

    plt.savefig(f'Main_Plots/perf-{rules[r_index]}.pdf', bbox_inches='tight')
    plt.close()

d_training = pd.read_pickle('training.pkl')
plot_line(d_training, 'Iteration', 'Perf', None, None, 'Model', f'Main_Plots/training.pdf',
          title=None, xlabel='Iterations', ylabel='Performance', hue_order=hue_perf)

plot_pie(rank['perf_last'], hue_pie, colors, f'Main_Plots/rank.pdf')
plot_pie(rank['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Main_Plots/rank_sub.pdf', circle=False)
plot_pie(mlp_rank['perf_last'], hue_pie, colors, f'Main_Plots/rank_mlp.pdf')
plot_pie(mha_rank_1['perf_last'], hue_pie, colors, f'Main_Plots/rank_mha.pdf')
plot_pie(rnn_rank['perf_last'], hue_pie, colors, f'Main_Plots/rank_rnn.pdf')
plot_pie(mlp_rank['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Main_Plots/rank_sub_mlp.pdf', circle=False)
plot_pie(mha_rank_1['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Main_Plots/rank_sub_mha.pdf', circle=False)
plot_pie(rnn_rank['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Main_Plots/rank_sub_rnn.pdf', circle=False)