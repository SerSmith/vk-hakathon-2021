import pandas as pd
from tqdm import tqdm
import numpy as np


def load_train(train_file):
    """Функция, загружающая файл

    Args:
        train_file (str): Путь к файлу

    Returns:
        pd.DataFrame: выборка
    """
    train = pd.read_csv(train_file)
    train_inv = train.copy()
    train_inv['tmp'] = train_inv['u']
    train_inv['u'] = train_inv['v']
    train_inv['v'] = train_inv['tmp']
    train_inv.drop(['tmp'], axis=1, inplace=True)
    train = pd.concat([train, train_inv])
    train.sort_values(['u'], ascending=False, inplace=True)
    return train


def get_users_to_predict(train_df):
    """Определение пользователей, для которых нужно сделать предсказание друзей

    Args:
        train_df pd.DataFrame: Датасет с данными для обучения

    Returns:
        pd.Series: Список пользователей, для которых надо сделать прогноз
    """
    users = train_df['u'].drop_duplicates()
    interesting_users = users[users % 8 == 1]
    return interesting_users


def k_fold_users(data, k):
    """Разбеение данных на части

    Args:
        data (pd.DataFrame): Данные для обучения
        k (int): количество кусков на которые надо разбить входные данные

    Returns:
        list(pd.Series): Фолды
    """
    out = []
    for i in range(k):
        out.append(data[data % k == i])
    return out


def predict_links(train_df, interesting_users):
    """Прогноз какая связь появится в ближайшее время

    Args:
        train_df (pd.DataFrame): Данные для обучения
        interesting_users (pd.Series): Пользователи для которых нужно сделать прогноз

    Returns:
        pd.DataFrame: Прогноз
    """
    friends_of_interesting_users = train_df[train_df['u'].isin(interesting_users)]
    friends_of_friends = friends_of_interesting_users.merge(train_df, left_on='v', right_on='u', suffixes=('', '_friends'))
    friends_of_friends = friends_of_friends.drop(columns=['v', 'u_friends'])
    friends_of_friends = friends_of_friends[friends_of_friends['u'] != friends_of_friends['v_friends']]
    friends_of_friends['weight'] = np.sqrt(np.log(friends_of_friends['h'])) * np.sqrt(np.log(friends_of_friends['h_friends']))
    weighted_sum = friends_of_friends.groupby(['u', 'v_friends']).agg({'weight': 'sum'})
    weighted_sum = weighted_sum.reset_index()
    weighted_sum['link'] = weighted_sum['u'].astype(str) + '_' + weighted_sum['v_friends'].astype(str)

    friends_of_interesting_users['link'] = friends_of_interesting_users['u'].astype(str) + '_' + friends_of_interesting_users['v'].astype(str)

    filtered_weighted_sum = weighted_sum[~ weighted_sum['link'].isin(friends_of_interesting_users['link'])]

    filtered_weighted_sum = filtered_weighted_sum[filtered_weighted_sum['v_friends'] % 2 == 1]

    filtered_weighted_sum = filtered_weighted_sum[filtered_weighted_sum['u'] < filtered_weighted_sum['v_friends']]

    filtered_weighted_sum = filtered_weighted_sum.sort_values(by='weight', ascending=False).groupby('u').head(10)
    return filtered_weighted_sum


def series2output(out, output_filname='output.txt'):
    """Запись результатов в txt файл

    Args:
        out (pd.DataFrame): Прогноз
        output_filname (str, optional): Файл, куда надо будет записать результат. Defaults to 'output.txt'.
    """
    aggregated = out[['u', 'v_friends']].astype(str).groupby('u').agg({'v_friends': lambda x: ','.join(x)}).reset_index()
    output = aggregated['u'].astype(str) + ": " + aggregated['v_friends'] + "\n"

    with open(output_filname, 'w') as h:
        h.writelines(output)


if __name__ == "__main__":
    train_df = load_train('data/train.csv')
    users = get_users_to_predict(train_df)

    folds = k_fold_users(users, 19)

    out = pd.DataFrame([], columns=['u',	'v_friends',	'weight'])

    for fold in tqdm(folds):
        res = predict_links(train_df, fold)
        res = res.drop(columns='link')
        out = out.append(res, ignore_index=True)

    series2output(out, output_filname='output_.txt')