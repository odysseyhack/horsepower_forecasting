import wget
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--region', type=str, default='se',
                    help='Region code')
parser.add_argument('--frequency', type=str, default='hourly',
                    help='frequency of the data')

args = parser.parse_args()

BASE_URL = f'https://www.nordpoolgroup.com/globalassets/marketdata-excel-files/'


def download_year(year, prognosis):
    if prognosis:
        production_url = BASE_URL + f'production-prognosis_{year}_{args.frequency}.xls'
        consumption_url = BASE_URL + f'consumption-prognosis-{args.region}_{year}_{args.frequency}.xls'
    else:
        production_url = BASE_URL + f'production-{args.region}-areas_{year}_{args.frequency}.xls'
        consumption_url = BASE_URL + f'consumption-{args.region}-areas_{year}_{args.frequency}.xls'

    print(f'...downloading from {production_url}')
    wget.download(production_url, '../data/')
    print(f'...downloading from {consumption_url}')
    wget.download(consumption_url, '../data/')

    return None


def download_data():
    for prognosis in [True, False]:
        for year in range(2015, 2020):
            print(f'Downloading data for year {year}')
            download_year(year, prognosis)
    return


if __name__ == "__main__":
    download_data()
