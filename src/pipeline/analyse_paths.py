from src.pipeline.Paths import Paths


def extract_path_events(datas, features, params, fps):
    paths = Paths()
    node_distance = params['node_distance']
    paths.create(datas, fps, node_distance)
    out_features = paths.analyse(features, params)
    return out_features
