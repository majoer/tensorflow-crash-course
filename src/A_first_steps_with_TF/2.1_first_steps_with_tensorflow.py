import california_housing_model

california_housing_model.train_model(
    learning_rate=0.00002,
    steps=10000,
    batch_size=5
)

plt.show()
