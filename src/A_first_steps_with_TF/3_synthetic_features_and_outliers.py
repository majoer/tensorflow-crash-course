from matplotlib import pyplot as plt
import california_housing_model as model

california_housing_dataframe = model.california_housing_dataframe

# Task 1
california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]

# Task 3
california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["rooms_per_person"].apply(lambda x: min(x, 5))

calibration_data = model.train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)

# Task 2
print("media_house_value description:")
print(california_housing_dataframe["median_house_value"].describe())

plt.subplot(2, 2, 3)
plt.ylabel("Predictions")
plt.xlabel("Targets")
plt.title("Predictions vs Targets")
plt.scatter(calibration_data["predictions"], calibration_data["targets"])


plt.subplot(2, 2, 4)
_ = california_housing_dataframe["rooms_per_person"].hist()

plt.show()
