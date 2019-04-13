import pandas as pd
import holoviews as hv
hv.extension('bokeh', 'matplotlib')


def read_data(path):
    df = pd.read_csv(path, index_col='Date')
    df.index = pd.DatetimeIndex(df.index)
    prefix = 'production_' if 'production' in path else 'consumption_'
    df = df.add_prefix(prefix)
    return df.sort_index(ascending=True)

def merge_data(production, consumption):
    df = production.join(consumption, on='Date')
    df.fillna(method='ffill', inplace=True)
    my_production = df['production_SE3'] + df['production_SE4']
    my_consumption = df['consumption_SE3']
    return my_production, my_consumption


def group_freq(df, freq):
    if freq == 'H':return df.groupby(df.index.hour).mean()
    if freq == 'D':return df.groupby(df.index.day).mean()
    if freq == 'M':return df.groupby(df.index.month).mean()
    if freq == 'Y':return df.groupby(df.index.year).mean()
    if freq == 'WD':return df.groupby(df.index.dayofweek).mean()


def visualize(prod, cons):
    prod = prod.hvplot.line(color='green', label='production')
    cons = cons.hvplot.line(color='red', label='consumption')
    return (prod * cons).opts(width=800, height=600)


production = read_data('../data/processed/production.csv')
consumption = read_data('../data/processed/consumption.csv')
production, consumption = merge_data(production, consumption)
