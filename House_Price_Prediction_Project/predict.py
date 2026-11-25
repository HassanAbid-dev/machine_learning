import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("house_model.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
keep = set(bundle["keep_societies"])
features = bundle["features"]


def predict_price(area_marla, bedrooms, bathrooms, society,
                  kitchens=2, store_rooms=1, servant_quarters=1, built_year=np.nan):
    """Return the predicted price in rupees."""
    if society not in keep:
        society = "Other"
    society_code = encoder.transform(pd.DataFrame({"Society": [society]}))[0][0]
    row = pd.DataFrame([{
        "Area_sqft": area_marla * 272.25,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Kitchens": kitchens,
        "Store Rooms": store_rooms,
        "Servant Quarters": servant_quarters,
        "Built Year": built_year,
        "Society": society_code,
    }])[features]
    return float(np.exp(model.predict(row))[0])


if __name__ == "__main__":
    price = predict_price(20, 5, 5, "DHA Defence")      # 1 kanal = 20 marla
    print(f"1 kanal, 5 bed, DHA Defence: Rs {price:,.0f} ({price / 1e7:.2f} crore)")
    price = predict_price(5, 3, 3, "Bahria Town")
    print(f"5 marla, 3 bed, Bahria Town: Rs {price:,.0f} ({price / 1e7:.2f} crore)")
    price = predict_price(5, 3, 3, "Some Unknown Society")
    print(f"5 marla, 3 bed, unknown society: Rs {price:,.0f} ({price / 1e7:.2f} crore)")