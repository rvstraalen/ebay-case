import pandas as pd
from datetime import datetime


# define constants
INPUT_FILENAME = './data/raw/case_dataset.csv'
OUTPUT_FILENAME = './data/interim/data_preprocessed.csv'

# read raw data set
dat = pd.read_csv(INPUT_FILENAME)

# Note: see notebooks/01-explore-data.ipynb for choices regarding features

# parse datetime from timestamp and create datetime features
# since all data is from the same single day, it's no use defining day/month/year-related features
dat['datetime'] = dat['timestamp'].apply(lambda x: datetime.strptime(str(x), '%y%m%d%H'))
dat = dat.assign(
    hour=dat['datetime'].dt.hour,
    is_working_hour=((dat['datetime'].dt.hour >= 8) & (dat['datetime'].dt.hour <= 18)),
    is_night_hour=((dat['datetime'].dt.hour >= 23) & (dat['datetime'].dt.hour <= 5))
)

# define categorical features
# N.B. The channel feature is also defined as categorical (see notebook)
categorical_features = ['channel', 'site_category', 'app_category', 'device_type', 'device_conn_type']
dat = dat.astype({c: 'category' for c in categorical_features})

# drop irrelevant columns
all_features = categorical_features + ['hour', 'is_working_hour', 'is_night_hour']
keep_columns = all_features+['is_clicked']
dat = dat.drop([c for c in dat.columns if c not in keep_columns], axis=1)

# quick and dirty one-hot-encoding
# not a sustainable solution if the model was to be used with future data
dat_ohe = pd.get_dummies(dat)

# save pre-processed data to disk
dat_ohe.to_csv(OUTPUT_FILENAME, index=False)

