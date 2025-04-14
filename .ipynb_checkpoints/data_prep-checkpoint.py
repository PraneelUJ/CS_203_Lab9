import mlrun
from sklearn.datasets import load_breast_cancer
import pandas as pd

@mlrun.handler(outputs=["dataset", "label_column"])
def dataset_loader(context, format="csv"):

    cancer = load_breast_cancer(as_frame=True)
    cancer_dataset = cancer.frame
    cancer_dataset['target'] = cancer.target
    
    context.logger.info('saving cancer dataset to {}'.format(context.artifact_path))
    context.log_dataset('cancer_dataset', df=cancer_dataset, format=format, index=False)
    
    return cancer_dataset, "target"

if __name__ == "__main__":
    with mlrun.get_or_create_ctx("cancer_generator", upload_artifacts=True) as context:
        dataset_loader(context, context.get_param("format", "csv"))