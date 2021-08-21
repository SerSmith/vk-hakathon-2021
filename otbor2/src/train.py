import pickle
import utils
import os
from data import InputDataTrain
from catboost import Pool, cv, CatBoostRegressor



# def train_model(data_class, algorithm_data_path="algorithm_data/"):
#     train = data_class.get_train()
#     education = data_class.get_education()
#     friends = data_class.get_friends()
#     graduations = utils.calcualte_graduations(train, education)

#     algorithm_data = {}
#     algorithm_data['train_data'] = train
#     algorithm_data['train_friends'] = friends
#     algorithm_data['graduations'] = graduations

#     with open(os.path.join(algorithm_data_path, "algorithm_data.pkl"), 'wb') as handle:
#         pickle.dump(algorithm_data, handle)

#     return train


def train_model(data_class, algorithm_data_path):
    params = {"iterations": 3000,
        'depth': 8,
        'l2_leaf_reg': 5,
        'learning_rate': 0.05,
        "loss_function": "RMSE",
        "verbose": True,
        "early_stopping_rounds": 10}

    train = data_class.get_train()
    train_no_age = train.drop(columns=['age', 'registered_year'])
    education = data_class.get_education()
    friends = data_class.get_friends()
    groups_data = data_class.get_groups()
    graduations = utils.calcualte_graduations(train, education)

    train_groups_quant = utils.calculate_quantity(train_no_age, groups_data)
    train_groups_quant = train_groups_quant[['uid', 'gid']].rename(columns={'gid': 'qroups_quant'})

    train_friends_quant = utils.calculate_quantity(train_no_age, friends)
    train_friends_quant = train_friends_quant[['uid', 'fuid']].rename(columns={'fuid': 'friends_quant'})

    group_dummies, top_groups, mlb = utils.calculate_group_dummies(train_no_age, groups_data)

    age_by_graduation = utils.calcualate_age_by_graduation(train_no_age, education, graduations)

    train_friens_age, mean_age_by_friends_dict = utils.calculate_friends_age(train_no_age, age_by_graduation, friends)

    mean_age_by_graduation = age_by_graduation['age_by_graduation'].mean()
    train_age_by_graduation = train_no_age.merge(age_by_graduation, on='uid', how='left')

    age_by_graduation['age_by_graduation'] = age_by_graduation['age_by_graduation'].fillna(mean_age_by_graduation)

    train_education_features = utils.add_education_quant_features(train_no_age, education)
    median_group_age = utils.calculate_median_group_age(train, groups_data)
    train_group_statistics = utils.calculate_group_statistics(train_no_age, groups_data, median_group_age)

    assert train.shape[0] ==\
        train_groups_quant.shape[0] ==\
        train_friends_quant.shape[0] ==\
        group_dummies.shape[0] ==\
        train_age_by_graduation.shape[0] ==\
        train_friens_age.shape[0] ==\
        train_education_features.shape[0] ==\
        train_group_statistics.shape[0]

    full_train = train.merge(train_groups_quant, on='uid').merge(train_friends_quant, on='uid').merge(group_dummies, on='uid').merge(train_age_by_graduation, on='uid').merge(train_friens_age, on='uid').merge(train_education_features, on='uid').merge(train_group_statistics, on='uid')


    cv_dataset = Pool(data=full_train.drop(columns=['age', 'uid']),
                    label=full_train['age'])
    scores = cv(cv_dataset,
                params,
                fold_count=4)

    params["iterations"] = scores['iterations'].max()

    cat = CatBoostRegressor(**params)

    full_train_to_model = full_train.drop(columns=['age', 'uid'])

    model = cat.fit(full_train_to_model, full_train['age'])

    algorithm_data = {}
    algorithm_data['graduations'] = graduations
    algorithm_data['mean_age_by_graduation'] = mean_age_by_graduation
    algorithm_data['top_groups'] = top_groups
    algorithm_data['mlb'] = mlb
    algorithm_data['dataset_columns'] = full_train_to_model.columns
    algorithm_data['model'] = model
    algorithm_data['mean_age_by_friends_dict'] = mean_age_by_friends_dict
    algorithm_data['train_data'] = train
    algorithm_data['train_friends'] = friends
    algorithm_data['median_group_age'] = median_group_age

    with open(os.path.join(algorithm_data_path, "algorithm_data.pkl"), 'wb') as handle:
        pickle.dump(algorithm_data, handle)
    
    return full_train


if __name__ == '__main__':
    data_class = InputDataTrain("./data")
    train_model(data_class, "./algorithm_data/")