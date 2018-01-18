from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import sys
TRAINING_DATA="school_data.csv"
TEST_DATA="school_test_data.csv"

def main(trainings=1):
    training_set=tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=TRAINING_DATA,
      target_dtype=np.int64,
      features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=TEST_DATA,
        target_dtype=np.int64,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[1024, 512, 256],

                                            model_dir="school_model",
                                            optimizer = tf.train.ProximalAdagradOptimizer(
                                                learning_rate=0.1,
                                                l1_regularization_strength=0.001
                                            )
                                            )

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    for i in range(trainings):
        # Train model.
        classifier.train(input_fn=train_input_fn, steps=100)
        accuracy_score = classifier.evaluate(input_fn=test_input_fn)
        print (accuracy_score)




    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)
    # estimator.train(input_fn=input_fn_train)
    # estimator.evaluate(input_fn=input_fn_eval)
    # estimator.predict(input_fn=input_fn_predict)

    # print("\nTest Accuracy: \n".format(accuracy_score))
    # print("ACcouracy ",accuracy_score)
    # Classify two new flower samples.
    new_samples = np.array(
        [
            [25,3.8,1],
            [26, 6.2, 2],
         [22,4.1,3]
         ], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    # print (predictions)
    predicted_classes = [p["predictions"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predicted_classes))


if __name__ == "__main__":
    params=sys.argv
    trainings=int(params[1]) if len(params) >1 else 1
    main(trainings)




