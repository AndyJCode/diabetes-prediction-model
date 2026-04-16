from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

print('Report methods:', [m for m in dir(Report) if not m.startswith('_')])
print('Preset methods:', [m for m in dir(DataDriftPreset) if not m.startswith('_')])
