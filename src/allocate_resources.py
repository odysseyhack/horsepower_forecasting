import pandas as pd
import holoviews as hv
import yaml
hv.extension('bokeh', 'matplotlib')
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def clean_data(df):
    df2 = df[['Charge_Point_ID', 'Volume']]
    df2['Start_datetime'], df2['Duration'] = pd.to_datetime(df['Start_datetime'], utc=True), df['Volume'] / 22
    df2['ConnectedDuration'] = pd.to_timedelta(df['Duration'].apply(lambda x: x.strftime('%H:%M:%S')))
    df2['ConnectedDuration'] = (df2['ConnectedDuration'].dt.total_seconds() / 3600).round(2)
    return df2


def feature_engg(df2, df):
    df2['weekday'] = df2['Start_datetime'].dt.weekday
    df2['hour'], df2['minute'] = df2['Start_datetime'].dt.hour, df2['Start_datetime'].dt.minute
    df2['hour'] = df2.apply(lambda x: x['hour'] + x['minute'] // 15 * 0.25, axis=1)
    df2['weekend'] = df2['weekday'].apply(lambda x: 1 if x > 4 else 0)
    return df2


def get_crosstab(df, row_axis='Charge_Point_ID', col_axis='hour', val_axis=None, normalize=False, aggfunc='mean'):
    temp_df = pd.crosstab(df[row_axis], df[col_axis], values=df[val_axis], normalize=normalize,
                          aggfunc=aggfunc).fillna(0).round(1)
    return temp_df


def plot_heatmap(val_axis, annot=False, save=True):
    plt.figure(figsize=(18, 24))
    heatmap = sns.heatmap(charging_consumption, cmap='magma', annot=annot)
    if save: plt.savefig(val_axis + '.png')
    plt.show()
    return heatmap


def get_preferences(df, hour):
    ct_df = get_crosstab(df, val_axis='Volume')
    preferences = ct_df[hour].sort_values(ascending=False).index
    return list(preferences)

def allocate_power(preferences, config, top=10):
    preferences = [station for station in preferences if station not in config['charged']]
    preferences = list(config['urgent']) + preferences
    return preferences[:top]


if __name__ == '__main__':
    df = pd.read_excel('../../pre_analysis/Datasets/Home Stations 1-44_NL.xlsx')
    df2 = clean_data(df)
    df2 = feature_engg(df2, df)

    with open('../configs/schedules.yaml', "r") as f:
        config = yaml.safe_load(f)

    preferences = get_preferences(df2, 23)
    print(allocate_power(preferences, config, 10))
