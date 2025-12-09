import numpy as np

# Crop-based agronomic climate rules
CROP_RULES = {
    "Cassava": {"min_rain": 600, "min_temp": 18, "max_temp": 32},
    "Maize": {"min_rain": 400, "min_temp": 15, "max_temp": 30},
    "Wheat": {"min_rain": 300, "min_temp": 10, "max_temp": 25},
    "Rice, paddy": {"min_rain": 800, "min_temp": 18, "max_temp": 32},
    "Soybeans": {"min_rain": 400, "min_temp": 15, "max_temp": 30},
    "Sweet potatoes": {"min_rain": 500, "min_temp": 18, "max_temp": 30},
    "Yams": {"min_rain": 700, "min_temp": 20, "max_temp": 32},
    "Potatoes": {"min_rain": 300, "min_temp": 8, "max_temp": 20},
    "Sorghum": {"min_rain": 250, "min_temp": 15, "max_temp": 32},
}

def apply_sanity_adjustment(crop, rainfall, temperature, raw_pred):
    rules = CROP_RULES.get(crop)
    if rules is None:
        return raw_pred, "No sanity adjustment applied."

    penalty = 1.0
    messages = []

    # Rainfall
    if rainfall < rules["min_rain"]:
        factor = max(rainfall / rules["min_rain"], 0.1)
        penalty *= factor
        messages.append("low rainfall")

    # Temperature low
    if temperature < rules["min_temp"]:
        factor = max(temperature / rules["min_temp"], 0.1)
        penalty *= factor
        messages.append("low temperature")

    # Temperature high
    if temperature > rules["max_temp"]:
        factor = max(rules["max_temp"] / temperature, 0.1)
        penalty *= factor
        messages.append("high temperature")

    corrected = raw_pred * penalty

    explanation = (
        "Sanity adjustment applied due to: " + ", ".join(messages)
        if messages else "Climate conditions suitable."
    )

    return corrected, explanation
