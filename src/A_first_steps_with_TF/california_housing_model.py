import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
california_housing_dataframe["median_house_value"] /= 1000.0
label = "median_house_value"
targets = california_housing_dataframe[label].astype('float32')


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="population"):

    print(california_housing_dataframe.head())

    periods = 10
    stepsInPeriod = steps / periods

    feature_data = california_housing_dataframe[[input_feature]].astype('float32')
    feature_columns = [tf.feature_column.numeric_column(input_feature)]

    def train_input_fn(): return input_fn(feature_data, targets, batch_size=batch_size)

    def prediction_input_fn(): return input_fn(feature_data, targets, num_epochs=1, shuffle=False)

    gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gd_optimizer = tf.contrib.estimator.clip_gradients_by_norm(gd_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=gd_optimizer
    )

    plt.figure(figsize=(15, 6))
    plt.subplot(2, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(label)
    plt.xlabel(input_feature)

    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[input_feature], sample[label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    root_mean_squared_errors = []

    for period in range(0, periods):

        _ = linear_regressor.train(
            input_fn=train_input_fn,
            steps=stepsInPeriod
        )

        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item["predictions"][0] for item in predictions])
        mean_squared_error = metrics.mean_squared_error(predictions, targets)
        root_mean_squared_error = math.sqrt(mean_squared_error)

        root_mean_squared_errors.append(root_mean_squared_error)
        print("Period: %02d Root mean squared Error (on training data): %0.3f" % (period, root_mean_squared_error))

        y_extents = np.array([0, sample[label].max()])
        weight = linear_regressor.get_variable_value("linear/linear_model/%s/weights" % input_feature)[0]
        bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[input_feature].max()), sample[input_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])

    plt.subplot(2, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    return calibration_data
