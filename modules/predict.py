import json
import dill
import os
import pandas as pd


def jsons_from_folder(dir_: str) -> list[dict]:
    """read all json files in given folder and add files content to list"""
    result = []
    for file in os.scandir(dir_):
        if not file.name.endswith('.json') or not file.is_file():
            continue
        with open(file) as json_file:
            result.append(json.load(json_file))
    return result


def predict():
    # определяем директорию проекта
    path = os.environ.get(
        'PROJECT_PATH',
        os.path.expanduser('~/PycharmProjects/33.6_homework')
    )

    # выбираем последнюю по времени модель в папке
    model_file = max(
        [file for file in os.listdir(f'{path}/data/models') if file.endswith('.pkl')]
    )

    # загружаем модель
    with open(f'{path}/data/models/{model_file}', 'rb') as file:
        model = dill.load(file)

    # формируем данные для предикта
    test_list = jsons_from_folder(f'{path}/data/test')
    test_df = pd.DataFrame.from_records(test_list)

    # предикт
    test_df['price_category'] = model.predict(test_df)

    # запись результатов в csv
    csv_file_name = f"prediction_{model_file.rstrip('.pkl').split('_')}.csv"
    test_df.to_csv(f"{path}/data/predictions/{csv_file_name}")


if __name__ == '__main__':
    predict()
