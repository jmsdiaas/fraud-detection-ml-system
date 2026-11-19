from fraudml.data import make_transactions
from fraudml.monitor import feature_drift
import json

df = make_transactions(4000, seed=42)
reference, recent = df.iloc[:2000], df.iloc[2000:]
print(json.dumps(feature_drift(reference, recent), indent=2))
