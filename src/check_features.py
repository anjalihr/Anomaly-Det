import pandas as pd
pd.set_option('display.width', 200)
df = pd.read_csv('/Users/anj/Projects/anomaly_det/data/features.csv')
cols = ['hour_zscore','geo_velocity_kmh','is_new_device','fingerprint_mismatch',
        'is_new_host','is_foreign_dept','graph_dist_from_history','failure_burst_count']
print(df.groupby('attack_type', dropna=False)[cols].mean().round(2))