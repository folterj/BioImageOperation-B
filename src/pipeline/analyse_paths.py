from src.pipeline.Paths import Paths


def extract_path_events(datas, features, params, general_params):
    paths = Paths()
    out_features = paths.run(datas, features, params, general_params)
    return out_features
