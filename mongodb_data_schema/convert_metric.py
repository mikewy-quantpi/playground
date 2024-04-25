import json
import uuid
import re

from json import JSONEncoder
from pymongo import MongoClient


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

def project_embedder_factory(raw_metric):
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

raw_metric = raw_metrics_dict['demographic_parity'][0]
project_embedder = project_embedder_factory(raw_metric)


class MetricEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return obj.__dict__

encoded_project_embedder = json.loads(json.dumps(
    obj=project_embedder,
    cls=MetricEncoder,
    indent=4
))

# print(encoded_project_embedder)
# print("=============================")
# print(type(encoded_project_embedder))
# print("=============================")
# insert_result = collection.insert_one(encoded_project_embedder)
# print(insert_result)


# performance embedder

# fairness settings

# perturber family

###############################################################################
# AssessmentResult
###############################################################################
class AssessmentResult:
    def __init__(self, subgroups, applicants, selected):
        self._id = uuid.uuid4()
        self.subgroups = subgroups
        self.applicants = applicants
        self.selected = selected


# assessment_result = AssessmentResult(
    # subgroups=["1", "2", "3"],
    # applicants=100,
    # selected=70
# )

# encoded_assessment_result = json.loads(json.dumps(
    # obj=assessment_result,
    # cls=MetricEncoder,
    # indent=4
# ))

# print(encoded_assessment_result)

class Metric:
    def __init__(self, name, type, projection_embedder, assessment_result):
        self.name = name
        self.type = type
        self.projection_embedder = projection_embedder
        self.assessment_result = assessment_result

###############################################################################
# TODO:
# - Create a function to read projection_embedder and convert assessment result
# - Refactor it into a factory method for Metric
# - Add perturber family
# - Add fairness settings
###############################################################################




###############################################################################
# insert into MongoDB
###############################################################################
# insert_result = collection.insert_one(metric.__dict__)
# print(insert_result)

client.close()
