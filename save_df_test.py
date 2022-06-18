# Import pandas library
import pandas as pd
import os

# initialize list elements
data = [10, 20, 30, 40, 50, 60]

# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['Numbers'])
home_dir = os.path.join('/home', 'yyu')

save_df_path = os.path.join(home_dir, 'test_dataframe_2.csv')

print(save_df_path)
df.to_csv(save_df_path, index=False)