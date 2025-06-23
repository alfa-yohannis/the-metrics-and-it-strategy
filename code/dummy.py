import pandas as pd

# Load the CSV file
df = pd.read_csv("universities2.csv")

# Extract unique university names
unique_universities = df["name"].unique()

# Save to CSV file
pd.DataFrame(unique_universities, columns=["university"]).to_csv("unique_universities.csv", index=False)

print("Saved unique universities to 'unique_universities.csv'")
