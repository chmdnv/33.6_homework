import json
from datetime import datetime

import dill
import os
import pandas as pd


path = os.environ.get('PROJECT_PATH', '~/PycharmProjects/33.6_homework')


def predict():
    model_file = max(
        [file for file in os.listdir(f'{path}/data/models') if file.endswith('.pkl')]
    )

    with open(model_file, 'rb') as file:
        model = dill.load(file)

    test = []

    for file in os.scandir(f'{path}/data/test'):
        if not file.name.endswith('.json') or not file.is_file():
            continue
        with open(file) as json_file:
            test.append(json.load(json_file))

    test_df = pd.DataFrame.from_records(test)

    test_df['price_category'] = model.predict(test_df)

    csv_file_name = f"prediction_{model_file.rstrip('.pkl').split('_')}.csv"
    test_df.to_csv(f"{path}/data/predictions/{csv_file_name}")


if __name__ == '__main__':
    predict()
