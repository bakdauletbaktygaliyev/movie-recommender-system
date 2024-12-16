
import pandas as pd

# Load the CSV file

# Read the original dataset
data = pd.read_csv("../data/ratings_with_users.csv")

# Select the required columns
user_data = data[['user_id', 'username', 'password']].drop_duplicates()

# Save the new file
user_data.to_csv("../data/user_credentials.csv", index=False)

