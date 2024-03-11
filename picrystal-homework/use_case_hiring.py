import argparse
import json
from functools import cached_property
import pandas as pd
import joblib

from picrystal_test import embedders

import picrystal_test.core
import picrystal_test.test_catalog
from picrystal_test.package_wrappers import PackageWrapper
from picrystal_test.embedders import BinaryEmbedderFromProbability


URL = 'https://storage.googleapis.com/picrystal-bucket/hiring/445b7773-3431-4c34-a762-ce8986670aa3_main_hiring_updated.csv'


class PositiveBinaryEmbedderFromProbability(BinaryEmbedderFromProbability):
    def __init__(self, on='predictions', threshold=0.5, tags=tuple()):
        super().__init__(on=on, threshold=threshold, tags=tags)

    def __call__(self, inputs):
        inputs, targets, predictions = inputs
        predictions = predictions[:, 1]  # only keep the prediction probablity for postive
        return super().__call__((inputs, targets, predictions))

    def info(self):
        return super().info()


class HiringUseCase:

    @cached_property
    def model(self):
        clf = joblib.load("hiring_model.joblib")
        return PackageWrapper(model=clf, package_name="sklearn", ml_case="classification")

    @cached_property
    def inputs(self):
        df = pd.read_csv(URL)
        df = df.drop("Unnamed: 0", axis=1)

        predictors = [
            "State",
            "Sex",
            "MaritalDesc",
            "CitizenDesc",
            "RaceDesc",
            "Department",
            "RecruitmentSource",
            "PerformanceScore",
            "SpecialProjectsCount"
        ]

        return df[predictors].values

    @cached_property
    def targets(self):
        df = pd.read_csv(URL)
        df = df.drop("Unnamed: 0", axis=1)

        target = "HiredOrNot"
        return df[target].values

    embedders = [
        embedders.LabelMapEmbedder(
            class_info=[
                {"default_class_value": 0, "value": 0, "name": "Hired"},
                {"default_class_value": 1, "value": 1, "name": "Not Hired"}
            ],
            tags=("predictions", "categorical", "binary", "hiring")
        ),

        embedders.BinaryEmbedder(
            on="groundtruth",
            tags=("categorical", "binary", "hiring")
        ),

        PositiveBinaryEmbedderFromProbability(
            on='predictions',
            threshold=0.7,
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=1,
            groups={1: "male", 2: "female"},
            tags=("sensitive", "gender", "sex"),
            name="gender"
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=2,
            groups={1: "married", 2: "single", 3: "divorced", 4: "widowed", 5: "separated"},
            tags=("marriage",),
            name="marriage"
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=3,
            groups={1: "US Citizen", 2: "Eligible NonCitizen", 3: "Non-Citizen"},
            tags=("citizenship",),
            name="citizenship"
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=4,
            groups={1: "white", 2: "black", 3: "two or more races", 4: "asian", 5: "native", 6: "hispanic"},
            tags=("sensitive", "race",),
            name="race",
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=5,
            groups={1: "engineering", 2: "sales", 3: "marketing", 4: "service"},
            tags=("department", ),
            name="department",
        )
    ]

    perturbers = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-spec", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    use_case = HiringUseCase()

    with open(args.metrics_spec) as f:
        metrics_spec = json.load(f)

    result = picrystal_test.core.run_all_tests(
        metrics_spec,
        use_case,
        picrystal_test.test_catalog.catalog
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)
