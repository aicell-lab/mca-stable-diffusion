import pandas as pd
import matplotlib.pyplot as plt

metric_path = "/data/wei/stable-diffusion/logs/2022-11-22T20-38-03_hpa-ldm-vq-4-hybrid-location/testtube/version_0/metrics.csv"

# load csv file into pandas DataFrame
df = pd.read_csv(metric_path)

# filter out empty rows based on val/loss_simple column
df = df[df['val/loss_simple'].notna()]

# create line plot
plt.plot( df['val/loss_simple'])
plt.xlabel('T')
plt.ylabel('Validation Loss (Simple)')
plt.title('Validation Loss vs T')

# save figure to a file
plt.savefig('./tmp.png')