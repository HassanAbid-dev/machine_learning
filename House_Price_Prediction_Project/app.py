import streamlit as st
from predict import predict_price, keep

st.set_page_config(page_title="Lahore House Price Estimator", page_icon="🏠")
st.title("🏠 Lahore House Price Estimator")
st.write("Enter the details of a house to get an estimated asking price.")


def format_price(rupees):
    if rupees >= 1e7:
        return f"{rupees / 1e7:.2f} crore"
    return f"{rupees / 1e5:.0f} lakh"


societies = sorted(keep) + ["Other (not listed)"]

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (marla)", min_value=2.0, max_value=100.0,
                           value=5.0, step=1.0, help="1 kanal = 20 marla")
    society = st.selectbox("Society / area", societies)
with col2:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=3)

if st.button("Estimate price"):
    price = predict_price(area, bedrooms, bathrooms, society)
    st.success(f"Estimated price: **{format_price(price)}** (Rs {price:,.0f})")
    if area >= 40:
        st.warning("This is a large house (2 kanal or more). The model has little data "
                   "on these, so treat the estimate as rough.")

st.caption("Based on asking prices from Zameen.com listings in 2025, not final sale prices. "
           "On test data the typical error was about 12%, and larger for luxury houses.")