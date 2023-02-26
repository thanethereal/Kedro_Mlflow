"""
This is a boilerplate pipeline
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=processing_data,
                inputs=["raw_data"],
                outputs=["dataset", "continuous_feature", "categorical_features"],
                name="processing_data",
            ),
            node(
                func=processing_data_encode,
                inputs=["dataset"],
                outputs="encode_data",
                name="processing_data_encode",
            ),
            node(
                func=feature_selection,
                inputs=["encode_data"],
                outputs="dataset_feature",
                name="feature_selection",
            ),
            node(
                func=train_test_split,
                inputs=["dataset_feature", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=train,
                inputs=["X_train", "y_train", "parameters"],
                outputs="sklearn_model",
                name="train",
            ),
            node(
                func=predict_on_test_data,
                inputs=["sklearn_model", "X_test"],
                outputs="y_pred",
                name="predict_on_test_data",
            ),
            node(
                func=predict_prob_on_test_data,
                inputs=["sklearn_model", "X_test"],
                outputs="y_pred_prob",
                name="predict_prob_on_test_data",
            ),
            node(
                func=get_metrics,
                inputs=["y_test", "y_pred", "y_pred_prob"],
                outputs="metrics",
                name="get_metrics",
            ),
            node(
                func=create_confusion_matrix_plot,
                inputs=["sklearn_model", "X_test", "y_test"],
                outputs=None,
                name="create_confusion_matrix_plot",
            ),
            node(
                func=create_experiment,
                inputs=["sklearn_model", "metrics", "parameters"],
                outputs=None,
                name="create_experiment",
            )
        ]
    )
