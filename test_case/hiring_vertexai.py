import argparse
import json
from functools import cached_property
from pathlib import Path
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from picrystal_metric_compute import embedders
#from picrystal_test import perturbers

import picrystal_metric_compute.core
import picrystal_metric_compute.metrics_catalog
from picrystal_metric_compute.package_wrappers import PackageWrapper

import numpy as np
from test import model


class HiringUseCase:
    target = 'HiredOrNot'
    predictors = [
        'State',
        'Sex',
        'MaritalDesc',
        'CitizenDesc',
        'RaceDesc',
        'Department',
        'RecruitmentSource',
        'PerformanceScore',
        'SpecialProjectsCount'
    ]
    class_info = [
            {'default_class_value': 0, 'value': 0, 'name': 'Hired'},
            {'default_class_value': 1, 'value': 1, 'name': 'Not Hired'}
    ]

    def __init__(self):
        self.train_df, self.val_df = train_test_split(
            self.df, test_size=0.2, random_state=2018, shuffle=True)

        self.model  # train the model

    @cached_property
    def df(self):
        df = pd.read_csv(
            'https://storage.googleapis.com/picrystal-bucket/hiring/445b7773-3431-4c34-a762-ce8986670aa3_main_hiring_updated.csv')
        df = df.drop('Unnamed: 0', axis=1)

        return df

    @cached_property
    def model(self):
        clf = model
        return PackageWrapper(model=clf,package_name="vertexAI", ml_case='classification')

    @cached_property
    def inputs(self):
        return self.val_df[self.predictors].values

    @cached_property
    def targets(self):
        return self.val_df[self.target].values

    embedders = [
        embedders.IdentityEmbedder(on='predictions', tags=('probabilities',)),
        embedders.LabelMapEmbedder(class_info=class_info, tags=('predictions','binary', 'categorical' )),

        embedders.BinaryEmbedder(on='groundtruth', tags=('binary','categorical')),

        embedders.CategoricalIdentityInputEmbedder(
            column=1,
            groups={1: 'male', 2: 'female'},
            tags=('sensitive', 'gender', 'sex'),
            name='gender'
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=2,
            groups={1: 'married', 2: 'single', 3: 'divorced', 4: 'widowed', 5: 'separated'},
            tags=('marriage',),
            name='marriage'
        ),
        embedders.CategoricalIdentityInputEmbedder(
            column=3,
            groups={1: 'US Citizen', 2: 'Eligible NonCitizen', 3: 'Non-Citizen'},
            tags=('sensitive', 'citizenship',),
            name='citizenship'
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=4,
            groups={1: 'white', 2: 'black or african american', 3: 'two or more races', 4: 'asian', 5: 'american indian or alaska native', 6: 'hispanic'}, 
            tags=('sensitive', 'race', ),
            name='race',
            missing_label=3,
        ),

    ]

    perturbers = [

    ]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--metrics-spec', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    usecase = HiringUseCase()
    #print(usecase.model(np.array([1,2,3,4,5,6,7,8,9]).reshape([-1,9])))

    with open(args.metrics_spec) as f:
        metrics_spec = json.load(f)

    result = picrystal_metric_compute.core.run_all_metrics(
        metrics_spec,
        usecase,
        picrystal_metric_compute.metrics_catalog.catalog
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
