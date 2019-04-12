from glob2 import glob
import pandas as pd
import os


def preprocess_data(df):
    features = [col for col in df.columns if 'SE' in col]
    df['Date'] = df['Date'] + '-' + df['Hours'].apply(lambda x: x[:2])
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y-%H')
    return df[features]


def read_xls(path):
    html_data = pd.read_html(path, header=2)[0]
    return pd.DataFrame(html_data)


def combine_files(pattern='production-se'):
    filepaths = [path for path in glob("../data/raw/*.xls") if pattern in path]
    df = pd.concat([read_xls(path) for path in filepaths], axis=0)
    return df.rename(columns={df.columns[0]: 'Date'})


def preprocess_and_save(pattern='production-se'):
    df = combine_files(pattern)
    df = preprocess_data(df)
    output_path = pattern.split('-')[0] + '_prognosis' if 'prog' in pattern else pattern.split('-')[0]
    df.to_csv(f'../data/processed/{output_path}.csv', index=False)
    return df

def test_preprocessing():
    assert os.path.exists('../data/processed/')
    assert len(glob('../data/raw/*.xls')) > 1

if __name__ == '__main__':
    test_preprocessing()
    for pattern in ['production-se', 'production-prog',
                    'consumption-se', 'consumption-prog'
                    ]:
        preprocess_and_save(pattern)
