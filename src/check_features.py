import os

import pandas as pd
pd.set_option('display.width', 200)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
cols = ['hour_zscore','geo_velocity_kmh','is_new_device','fingerprint_mismatch',
        'is_new_host','is_foreign_dept','graph_dist_from_history','failure_burst_count']
print(df.groupby('attack_type', dropna=False)[cols].mean().round(2))