from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(predict_params):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]
    classifier = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                           hidden_units=[1024, 512, 256],

                                           model_dir="school_model",
                                           optimizer=tf.train.ProximalAdagradOptimizer(
                                               learning_rate=0.1,
                                               l1_regularization_strength=0.001
                                           )
                                           )
    new_samples = np.array(
        [
            predict_params,
        ], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)
    predictions = list(classifier.predict(input_fn=predict_input_fn))

    predicted_classes = [p["predictions"] for p in predictions]

    return predicted_classes[0][0]
    # print(
    #     "New Samples, Class Predictions:    {}\n"
    #         .format(predicted_classes[0][0]))


if __name__ == "__main__":
    parameters=sys.argv
    # print(parameters)
    if len(parameters) < 4:
        print("Error some fields missing.")
    else:
        prs=[float(d) for d in parameters[1:]]
        # print(prs)
        prediction=main(prs)
        print( prs, " => " , prediction)

