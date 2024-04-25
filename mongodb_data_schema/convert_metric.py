import json
import uuid
import re

from json import JSONEncoder
from pymongo import MongoClient
from itertools import product


client = MongoClient('mongodb://localhost:27017/')
db = client['playground']
collection = db['metric']

###############################################################################
# Load raw metrics file
###############################################################################
raw_metrics_file = './llm_microsoft_phi_2_prompt_option_0_api_option_1.json'
with open(raw_metrics_file, 'r') as file:
    raw_metrics_dict = json.load(file)


###############################################################################
# Convert and insert one demographic parity metric as an example
###############################################################################
# projection embedder
class ProjectionEmbedder:
    def __init__(self, name, dimensions):
        self.name = name
        self.dimensions = dimensions

class Dimension:
    def __init__(self, index, name, properties, bins):
        self.index = index
        self.name = name
        self.properties = properties
        self.bins = bins

class Bin:
    def __init__(self, index, name):
        self._id = uuid.uuid4()
        self.index = index
        self.name = name

def projection_embedder_factory(raw_metric):
    dimensions = []
    dim_index = -1
    for key in raw_metric.keys():
        if not re.match(r'feature', key):
            continue
        dim_index += 1
        dim_name = raw_metric[key]['name']
        dim_properties = {}  # TODO: leave it empty now
        dim_bins = []
        for bin in raw_metric[key]['class_info']:
            dim_bins.append(Bin(bin['value'], bin['name']))
        dimensions.append(Dimension(dim_index, dim_name, dim_properties, dim_bins))
    return ProjectionEmbedder(
        name=', '.join([dim.name for dim in dimensions]),
        dimensions=dimensions
    )


class MetricEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return obj.__dict__

# raw_metric = raw_metrics_dict['demographic_parity'][0]
# projection_embedder = projection_embedder_factory(raw_metric)

# encoded_projection_embedder = json.loads(json.dumps(
    # obj=projection_embedder,
    # cls=MetricEncoder,
    # indent=4
# ))

###############################################################################
# AssessmentResult
###############################################################################
class AssessmentResult:
    def __init__(self, subgroup, applicants, selected):
        self._id = uuid.uuid4()
        self.subgroup = subgroup
        self.applicants = applicants
        self.selected = selected

class Metric:
    def __init__(self, name, type, projection_embedder, assessment_results):
        self.name = name
        self.type = type
        self.projection_embedder = projection_embedder
        self.assessment_results = assessment_results

###############################################################################
# TODO:
# - Create a function to read projection_embedder and convert assessment result
# - Refactor it into a factory method for Metric
# - Add perturber family
# - Add fairness settings
###############################################################################
def assessment_results_factory(raw_metric):
    embedder = projection_embedder_factory(raw_metric)

    bin_ids = [[bin._id for bin in dim.bins] for dim in embedder.dimensions]
    subgroups = list(product(*bin_ids))
    applicants = raw_metric['metric']['# of applicants']
    selected = raw_metric['metric']['# of selected']

    results = list(zip(subgroups, applicants, selected))
    assessment_results = [AssessmentResult(a, b, c) for a, b, c in results]

    return embedder, assessment_results


raw_metric = raw_metrics_dict['demographic_parity_mvariable'][0]
projection_embedder, assessment_results = assessment_results_factory(raw_metric)

metric = Metric(
    name="mike",
    type="performance",
    projection_embedder=projection_embedder,
    assessment_results=assessment_results
)

encoded_metric = json.loads(json.dumps(
    obj=metric,
    cls=MetricEncoder,
    indent=4
))
print(encoded_metric)


###############################################################################
# insert into MongoDB
###############################################################################
insert_result = collection.insert_one(encoded_metric)
print(insert_result)

client.close()
