import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Load the dataset
df = pd.read_csv("total-ghg-emissions.csv")

# Preview first few rows
df.head()

print(df.isnull().sum())

# Fill missing values with the median
df.fillna(df.median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)
# Group data by year and sum emissions
yearly_emissions = df.groupby("Year")["Total_Emissions"].sum()

# Display summary
print(yearly_emissions)
plt.figure(figsize=(10,5))
plt.plot(df["Year"], df["Total_Emissions"], marker="o", linestyle="-", color="b")

# Customize plot
plt.xlabel("Year")
plt.ylabel("Total GHG Emissions")
plt.title("Global GHG Emissions Over Time")
plt.grid(True)

plt.show()

# Prepare data for regression model
X = df["Year"].values.reshape(-1, 1)
y = df["Total_Emissions"].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for future years
future_years = np.array([2025, 2030, 2035]).reshape(-1, 1)
predictions = model.predict(future_years)

# Print predictions
print("Predicted emissions for 2025, 2030, 2035:", predictions)