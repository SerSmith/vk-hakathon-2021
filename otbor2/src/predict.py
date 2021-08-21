import os
import pickle
import utils
from data import InputDataTest


def predict_model(data_class, algorithm_data_path="./otbor2/algorithm_data/", result_save_path='/var/log/result'):
    
    test = data_class.get_test()

    test_no_reg_date = test.drop(columns='registered_year')
 
    education = data_class.get_education()
    friends = data_class.get_friends()
    groups_data = data_class.get_groups()

    with open(os.path.join(algorithm_data_path, "algorithm_data.pkl"), 'rb') as handle:
        algorithm_data = pickle.load(handle)
    
    graduations = algorithm_data['graduations']
    top_groups = algorithm_data['top_groups']
    mean_age_by_graduation = algorithm_data['mean_age_by_graduation']
    mean_age_by_friends_dict = algorithm_data['mean_age_by_friends_dict']
    model = algorithm_data['model']
    dataset_columns = algorithm_data['dataset_columns']
    train_data = algorithm_data['train_data']
    friends_train = algorithm_data['train_friends']
    median_group_age = algorithm_data['median_group_age']

    test_groups_quant = utils.calculate_quantity(test_no_reg_date, groups_data)
    test_groups_quant = test_groups_quant[['uid', 'gid']].rename(columns={'gid': 'qroups_quant'})

    test_friends_quant = utils.calculate_quantity(test_no_reg_date, friends)
    test_friends_quant = test_friends_quant[['uid', 'fuid']].rename(columns={'fuid': 'friends_quant'})

    group_dummies, _, _ = utils.calculate_group_dummies(test_no_reg_date, groups_data, top_groups)

    age_by_graduation = utils.calcualate_age_by_graduation(test_no_reg_date, education, graduations)

    age_by_graduation['age_by_graduation'] = age_by_graduation['age_by_graduation'].fillna(mean_age_by_graduation)

    friends_appended = friends.append(friends_train, ignore_index=True).drop_duplicates()

    age_by_graduation_appended = utils.combine_age_files(train_data[['uid', 'age']].rename(columns={'age': 'age_by_graduation'}), age_by_graduation)

    test_friens_age, _ = utils.calculate_friends_age(test_no_reg_date, age_by_graduation_appended, friends_appended, mean_age_by_friends_dict)

    test_age_by_graduation = test_no_reg_date.merge(age_by_graduation, on='uid', how='left')

    test_age_by_graduation['age_by_graduation'] = test_age_by_graduation['age_by_graduation'].fillna(mean_age_by_graduation)

    test_education_features = utils.add_education_quant_features(test_no_reg_date, education)

    test_group_statistics = utils.calculate_group_statistics(test_no_reg_date, groups_data, median_group_age)

    assert test.shape[0] ==\
           test_groups_quant.shape[0] ==\
           test_friends_quant.shape[0] ==\
           group_dummies.shape[0] ==\
           test_age_by_graduation.shape[0] ==\
           test_friens_age.shape[0] ==\
           test_education_features.shape[0] ==\
           test_group_statistics.shape[0]

    full_test = test.merge(test_groups_quant, on='uid').merge(test_friends_quant, on='uid').merge(group_dummies, on='uid').merge(test_age_by_graduation, on='uid').merge(test_friens_age, on='uid').merge(test_education_features, on='uid').merge(test_group_statistics, on='uid')

    output_ = full_test[['uid']]
    output_.loc[:, 'age'] = model.predict(full_test[dataset_columns])
    output_.to_csv(result_save_path, index=False)

    return None


if __name__ == '__main__':
    data_class = InputDataTest('/tmp/data/')
    predict_model(data_class)
