"""Shared feature contract; deliberately excludes IDs and post-event signals."""
FEATURES = [
    "amount", "hour", "account_age_days", "transactions_24h", "avg_amount_30d",
    "distance_km", "device_new", "ip_risk_score", "merchant_risk_score",
]
TARGET = "is_fraud"
TIME = "event_time"
