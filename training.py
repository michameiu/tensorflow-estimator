#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import sys
from sys import stdout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

TRAINING_DATA="school_data.csv"
TEST_DATA="school_test_data.csv"
MODEL_DIR="trained_model_09"
EXPORT_MODEL_DIR="%s/export"%(MODEL_DIR)

def main(trainings=10,export=False):

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

    # Build 3 layer DNN with 1024,512,256 units respectively.
    classifier = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[1024, 512, 256],
                                            # activation_fn=tf.nn.relu,
                                            model_dir=MODEL_DIR,
                                           dropout=0.9,
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
        perc=(float(i+1)/float(trainings))*100
        classifier.train(input_fn=train_input_fn, steps=5000)
        # if (i+5)%5 == 0:
        accuracy_score = classifier.evaluate(input_fn=test_input_fn)
        # print (accuracy_score)
        ac=accuracy_score
        stdout.write("\r Training:%.0f%s   Avg Loss: %s   TotalSteps: %s    Loss: %s"
                     %(perc,"%",ac["average_loss"],ac["global_step"],ac["loss"]))
        stdout.flush()
    print ()



    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)
    # estimator.train(input_fn=input_fn_train)
    # estimator.evaluate(input_fn=input_fn_eval)
    # estimator.predict(input_fn=input_fn_predict)

    # print("\nTest Accuracy: \n".format(accuracy_score))
    # print("ACcouracy ",accuracy_score)
    # Classify two new flower samples.
    samples=[[25,3.8,1],
            [26, 6.2, 2],
         [22,4.1,3]]
    new_samples = np.array(
        samples, dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    # print (predictions)
    predicted_classes = [p["predictions"][0] for p in predictions]

    print(
        "Predictions: {} == > {}\n"
            .format( samples, predicted_classes))


    ###Exporting the model

    def save_model():
        print ("Saving Model .....")
        serv_feature_spec =  {"x":tf.FixedLenFeature(shape=[3],dtype=np.float32)} #[tf.FixedLenFeature("x", shape=[3]) ]# .numeric_column(]
        serving_fn=tf.estimator.export.build_parsing_serving_input_receiver_fn(serv_feature_spec)
        classifier.export_savedmodel(export_dir_base=EXPORT_MODEL_DIR,
                                     serving_input_receiver_fn=serving_fn
                                     )
    if export:
        save_model()


if __name__ == "__main__":
    params=sys.argv
    # print(parameters)
    # print( params[1] , len(params), params[1])
    if len(params) < 2:
        raise AssertionError("Argument not provide")

    if params[1].isdigit() :
        trainings=int(params[1]) if len(params) >1 else 1
        main(trainings,export=False)
    elif type(params[1]) == str and params[1] == "export":
        main(1,export=True)
    else:
        print ("error")





