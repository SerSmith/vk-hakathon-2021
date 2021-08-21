import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


class InputData():

    def __init__(self, path_to_data):
        self.PATH_TO_DATA = path_to_data
        self.data = None
        self.friends = None
        self.education = None
        self.groups = None
        self.test = None

    def _get_data(self, file_name='train.csv'):
        if self.data is None:
            self.data = pd.read_csv(os.path.join(self.PATH_TO_DATA, file_name))
        return self.data

    def _get_friends(self, file_name='friends.csv'):
        if self.friends is None:
            self.friends = pd.read_csv(os.path.join(self.PATH_TO_DATA, file_name))
            self.friends = self.friends.append(self.friends.rename(columns={'uid': 'fuid', 'fuid': 'uid'}), ignore_index=True)
            self.friends = self.friends.drop_duplicates()
        return self.friends

    def _get_education(self, file_name="trainEducationFeatures.csv"):
        MIN_DATE = 1960
        if self.education is None:
            self.education = pd.read_csv(os.path.join(self.PATH_TO_DATA, file_name))
            self.education['uid'] = self.education['uid'].astype(int)
            self.education = self.education.rename(columns={'school_education': 'graduation_0'})
            for i in range(8):
                self.education.loc[self.education[f'graduation_{i}'] <= MIN_DATE, f'graduation_{i}'] = np.nan
        return self.education

    def _get_groups(self, file_name="trainGroups.csv"):
        if self.groups is None:
            self.groups = pd.read_csv(os.path.join(self.PATH_TO_DATA, file_name))
        return self.groups
    
    def get_data(self):
        return self._get_data(file_name='train.csv')

    def get_friends(self):
        return self._get_friends(file_name='friends.csv')

    def get_education(self):
        return self._get_education(file_name="trainEducationFeatures.csv")

    def get_groups(self):
        return self._get_groups(file_name="trainGroups.csv")


class InputDataValidate(InputData):
    TEST_PERC = 0.001

    def __init__(self, path_to_data):
        self.train = None
        self.test = None
        self.true_test_target = None
        super().__init__(path_to_data)


    @staticmethod
    def __create_train_test(data, test_perc):
        train, test = train_test_split(data, test_size=test_perc)
        true_test_target = test[['uid', 'age']]
        test = test.drop(columns=['age'])
        return train, test, true_test_target

    def get_train(self):
        if self.data is None:
            self.get_data()

        if self.train is None:
            self.train, self.test, self.true_test_target = InputDataValidate.__create_train_test(self.data, self.TEST_PERC)
        return self.train

    def get_test(self):
        if self.data is None:
            self.get_data()

        if self.test is None:
            self.train, self.test, self.true_test_target = InputDataValidate.__create_train_test(self.data, self.TEST_PERC)
        return self.test

    def get_true_test_target(self):
        if self.data is None:
            self.get_data()

        if self.true_test_target is None:
            self.train, self.test, self.true_test_target = InputDataValidate.__create_train_test(self.data, self.TEST_PERC)
        return self.true_test_target


class InputDataTrain(InputData):

    def __init__(self, path_to_data):
        self.train = None
        super().__init__(path_to_data)

    def get_train(self):
        if self.train is None:
            self.train = self.get_data()

        return self.train


class InputDataTest(InputData):
    def __init__(self, path_to_data):
        self.test = None
        super().__init__(path_to_data)

    def get_test(self):

        if self.test is None:
            self.test = self.get_data()
        return self.test

    def get_data(self):
        return self._get_data(file_name='test.csv')

    def get_friends(self):
        return self._get_friends(file_name='friends.csv')

    def get_education(self):
        return self._get_education(file_name="testEducationFeatures.csv")

    def get_groups(self):
        return self._get_groups(file_name="testGroups.csv")
