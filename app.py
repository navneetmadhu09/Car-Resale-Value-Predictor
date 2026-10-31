
import streamlit as st
import pickle as pkl

# Page configuration
st.set_page_config(
    page_title="Car Resale Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load saved files
lbl = pkl.load(open('resalecarencoder.sav', 'rb'))
model = pkl.load(open('resalecar.sav', 'rb'))
scaler = pkl.load(open('resalecarscaler.sav', 'rb'))


# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: #101116;
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.main-title {
    text-align: center;
    color: #ff5252;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #a5a5a5;
    font-size: 16px;
    margin-bottom: 35px;
}

.section-title {
    color: #ff5252;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #ff4545;
    color: white;
    border: none;
    border-radius: 8px;
    height: 48px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #d93636;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background-color: #20222b;
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #ff5252;
}

[data-testid="stMetricValue"] {
    color: #ff5252;
}

div[data-testid="stForm"] {
    background-color: #191b23;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #30323c;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">🚗 CAR RESALE VALUE PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict your car resale value using Machine Learning</div>',
    unsafe_allow_html=True
)


# Input form
with st.form("car_form"):

    # Car details
    st.markdown(
        '<div class="section-title">🚘 Car Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Car Brand", placeholder="Eg: Maruti")

        year = st.number_input(
            "Manufacturing Year",
            min_value=1900,
            max_value=2030,
            value=2019
        )

        ft = st.selectbox(
            "Fuel Type",
            ['CNG', 'Diesel', 'Petrol']
        )

        owntyp = st.selectbox(
            "Owner Type",
            ['First', 'Second']
        )

    with col2:
        location = st.text_input(
            "Location",
            placeholder="Eg: Mumbai"
        )

        kmd = st.number_input(
            "Kilometers Driven",
            min_value=0,
            value=10000
        )

        tr = st.selectbox(
            "Transmission",
            ['Automatic', 'Manual']
        )

    st.divider()

    # Technical specifications
    st.markdown(
        '<div class="section-title">⚙️ Technical Specifications</div>',
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2)

    with col3:
        mil = st.number_input(
            "Mileage",
            min_value=0.0,
            value=18.0
        )

        eng = st.number_input(
            "Engine Capacity (CC)",
            min_value=0.0,
            value=998.0
        )

    with col4:
        pow = st.number_input(
            "Maximum Power (BHP)",
            min_value=0.0,
            value=50.0
        )

        seat = st.number_input(
            "Number of Seats",
            min_value=1,
            value=5
        )

    st.divider()

    predict = st.form_submit_button(
        "🚀 PREDICT RESALE VALUE"
    )


# Prediction
if predict:

    if not brand.strip() or not location.strip():
        st.warning("Please enter the car brand and location.")

    else:
        try:
            br = lbl[0].transform([brand.strip()])[0]
            loc = lbl[1].transform([location.strip()])[0]
            fuel = lbl[2].transform([ft])[0]
            trans = lbl[3].transform([tr])[0]
            owner = lbl[4].transform([owntyp])[0]

            features = [[
                br, loc, year, kmd, fuel,
                trans, owner, mil, eng, pow, seat
            ]]

            scaled_features = scaler.transform(features)

            result = model.predict(scaled_features)[0]

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Prediction Result</div>',
                unsafe_allow_html=True
            )

            st.metric(
                label="Estimated Car Resale Value",
                value=f"{result:.2f}"
            )

            st.success("Prediction completed successfully!")

            st.caption(
                "Note: The predicted value is an estimate "
                "based on the trained machine learning model."
            )

        except ValueError as e:
            st.error(
                "Invalid input. Please check the brand and "
                "location spelling and ensure the categories "
                "exist in the training dataset."
            )


# Footer
st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#777;padding:15px;">
        CAR RESALE VALUE PREDICTOR | Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
