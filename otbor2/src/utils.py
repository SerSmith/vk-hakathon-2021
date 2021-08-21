import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def calcualte_graduations(train, education):
    train = train.merge(education, on='uid')

    for i in range(8):
        train[f'graduation_age_{i}'] = train['age'] - (2021 - train[f'graduation_{i}'])

    graduations = {i:  
                   {'std': train[f'graduation_age_{i}'].std(),
                    'mean': train[f'graduation_age_{i}'].mean()} for i in range(8)}
    return graduations


def calcualate_age_by_graduation(data, education, graduations):
    data = data.merge(education, on='uid')

    for i in range(8):
        data[f'age_by_graduation_{i}'] = graduations[i]['mean'] + (2021 - data[f'graduation_{i}'])
        data[f'weight_by_graduation_{i}'] = 1 / graduations[i]['std']
        data.loc[data[f'age_by_graduation_{i}'].isna(), f'weight_by_graduation_{i}'] = np.nan

    data['sum_weights'] = data[[f'weight_by_graduation_{i}' for i in range(8)]].fillna(0).sum(axis=1)

    data['age_by_graduation'] = data.apply(lambda x: np.nansum([x[f'age_by_graduation_{i}'] * x[f'weight_by_graduation_{i}'] / x['sum_weights'] for i in range(8)]), axis=1)

    return data.loc[data['sum_weights'] != 0, ['uid', 'age_by_graduation']]


def calculate_age_by_friends(age_by_graduation, friends):
    frieands_age_by_graduation = age_by_graduation.merge(friends, on='uid').drop(columns='uid').rename(columns={'fuid': 'uid'})
    frieands_age_by_graduation = frieands_age_by_graduation.groupby('uid').mean().reset_index()
    return frieands_age_by_graduation


def combine_age_files(file1, file2):
    return file1.append(file2, ignore_index=True).drop_duplicates(subset=["uid"], keep="first")


def calculate_quantity(data, quantity_data):
    data_quant = quantity_data.groupby('uid').count().reset_index()
    out = data.merge(data_quant, on='uid', how='left')
    out = out.fillna(0)
    return out


def calculate_group_dummies(data, groups_data, top_groups_=None, mlb_=None, quantity_top_groups=1000):

    if top_groups_ is None:
        top_groups = groups_data['gid'].value_counts().head(quantity_top_groups).index
    else:
        top_groups = top_groups_

    groups_data_limited = groups_data[groups_data['gid'].isin(top_groups)]
    gids_in_lists = groups_data_limited.groupby('uid').agg({'gid': list}, axis=0)
    
    if mlb_ is None:
        mlb = MultiLabelBinarizer(sparse_output=True)

        groups_dummies = gids_in_lists.join(pd.DataFrame.sparse.from_spmatrix(
                                            mlb.fit_transform(gids_in_lists.pop('gid')),
                                            index=gids_in_lists.index,
                                            columns=mlb.classes_))
    else:
        mlb = mlb_
        groups_dummies = gids_in_lists.join(pd.DataFrame.sparse.from_spmatrix(
                                    mlb.transform(gids_in_lists.pop('gid')),
                                    index=gids_in_lists.index,
                                    columns=mlb.classes_))
    # groups_dummies = pd.get_dummies(groups_data_limited, columns=['gid'])
    # if top_groups_ is not None:
    #     not_presented_columns = list(set(top_groups_) - set(groups_dummies.columns))
    #     groups_dummies[not_presented_columns] = 0

    out = data.merge(groups_dummies, how='left', on='uid')
    out = out.fillna(0)
    return out, top_groups, mlb


def calculate_friends_age(data, age_by_graduation, friends, mean_age_by_friends_dict_=None, num_itterations=10):
    
    known_age = age_by_graduation.copy()
    known_age = known_age.rename(columns={'age_by_graduation': 'age'})

    if mean_age_by_friends_dict_ is None:
        mean_age_by_friends_dict = {}
    else:
        mean_age_by_friends_dict = mean_age_by_friends_dict_

    for i in range(1, num_itterations + 1):
        age_by_friends = calculate_age_by_friends(known_age, friends)
        known_age = combine_age_files(known_age, age_by_friends)
        data = data.merge(age_by_friends, on='uid', how='left')
        data = data.rename(columns={'age': f'age_by_friends{i}'})
        if mean_age_by_friends_dict_ is None:
            mean_age_by_friends_dict[i] = data[f'age_by_friends{i}'].mean()

        data[f'age_by_friends{i}'] = data[f'age_by_friends{i}'].fillna(mean_age_by_friends_dict[i])
    
    return data, mean_age_by_friends_dict


def add_education_quant_features(data, education):
    stats = education[['uid']]
    for i in range(8):
        stats.loc[:, f'has_graduation_{i}'] = ~education[f'graduation_{i}'].isna()

    stats.loc[:, 'quant_graduation'] = stats[[f'has_graduation_{i}' for i in range(8)]].sum(axis=1)
    stats.loc[:, 'quant_graduation_higher'] = stats[[f'has_graduation_{i}' for i in range(1, 8)]].sum(axis=1)
    out = data[['uid']].merge(stats, on='uid')
    out = out.fillna(0)
    return out


def calculate_median_group_age(train, groups, min_group_size=10):
    users_groups = train.merge(groups, on='uid')
    gid_counts_dict = groups['gid'].value_counts()
    gid_popular = gid_counts_dict[gid_counts_dict >= min_group_size].index
    only_popular = users_groups[users_groups['gid'].isin(gid_popular)]
    median_group_age = only_popular.groupby('gid').aggregate({'age': 'median'}).reset_index()
    return median_group_age


def calculate_group_statistics(data, groups, median_age):
    user_groups = data.merge(groups, on='uid')
    group_statistics = user_groups.merge(median_age, on='gid').groupby('uid').agg({'age': ['min', 'max', 'mean', 'std']})
    group_statistics.columns = ['_'.join([column_name1, column_name2]) for column_name1, column_name2 in group_statistics.columns]
    group_statistics = group_statistics.reset_index()
    group_statistics = data.merge(group_statistics, on='uid', how='left')
    group_statistics = group_statistics.fillna(0)
    return group_statistics
    