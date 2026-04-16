from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/raw/heart_combined.csv')
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df, current_data=df)
    items = list(report.items())
    print('items count:', len(items))
    for item in items:
        print('item:', type(item), item)