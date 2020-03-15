from DataPipeline import DataPipeline

def train_model(pipeline_cfg):
    data_pipeline = DataPipeline(
        **pipeline_cfg
    )