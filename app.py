import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("return joblib.load("Model/hotel_cancellation_model.pkl")

st.set_page_config(page_title="Hotel Cancellation Risk", page_icon="🏨")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Predict the probability of a booking being cancelled.")

try:
    model = load_model()
    st.success("Model loaded successfully.")
except:
    st.error("Model file not found.")
    st.stop()

st.divider()
st.subheader("📋 Booking Information")

col1, col2 = st.columns(2)

# COLUMN 1 INPUTS
with col1:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])

    lead_time = st.number_input("Lead Time (days)", 0, 365, 30)

    meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC"])

    market_segment = st.selectbox(
        "Market Segment",
        ["Online TA", "Offline TA/TO", "Direct", "Groups",
         "Corporate", "Complementary", "Aviation"]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO", "Direct", "Corporate", "GDS"]
    )

    is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])

    previous_cancellations = st.number_input(
        "Previous Cancellations", 0, 20, 0
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous Bookings Not Canceled", 0, 20, 0
    )

# COLUMN 2 INPUTS
with col2:
    reserved_room_type = st.selectbox(
        "Reserved Room Type",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    assigned_room_type = st.selectbox(
        "Assigned Room Type",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    booking_changes = st.number_input("Booking Changes", 0, 10, 0)

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"]
    )

    days_in_waiting_list = st.number_input(
        "Days in Waiting List", 0, 365, 0
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient", "Transient-Party", "Contract", "Group"]
    )

    adr = st.number_input("Average Daily Rate (ADR)", 0.0, 1000.0, 100.0)

    required_car_parking_spaces = st.number_input(
        "Required Parking Spaces", 0, 5, 0
    )

    total_of_special_requests = st.number_input(
        "Total Special Requests", 0, 5, 0
    )

    total_stays = st.number_input("Total Stays (nights)", 1, 30, 1)

    total_guests = st.number_input("Total Guests", 1, 10, 2)

    has_agent = st.selectbox("Has Agent?", [0, 1])

    has_company = st.selectbox("Has Company?", [0, 1])

    room_assigned_different = st.selectbox(
        "Room Assigned Different?", [0, 1]
    )

    arrival_month_num = st.number_input(
        "Arrival Month (1-12)", 1, 12, 6
    )

    lead_time_category = st.selectbox(
        "Lead Time Category",
        ["last_minute", "short", "medium", "long"]
    )


# DATAFRAME
input_data = pd.DataFrame([{
    'hotel': hotel,
    'lead_time': lead_time,
    'meal': meal,
    'market_segment': market_segment,
    'distribution_channel': distribution_channel,
    'is_repeated_guest': is_repeated_guest,
    'previous_cancellations': previous_cancellations,
    'previous_bookings_not_canceled': previous_bookings_not_canceled,
    'reserved_room_type': reserved_room_type,
    'assigned_room_type': assigned_room_type,
    'booking_changes': booking_changes,
    'deposit_type': deposit_type,
    'days_in_waiting_list': days_in_waiting_list,
    'customer_type': customer_type,
    'adr': adr,
    'required_car_parking_spaces': required_car_parking_spaces,
    'total_of_special_requests': total_of_special_requests,
    'total_stays': total_stays,
    'total_guests': total_guests,
    'has_agent': has_agent,
    'has_company': has_company,
    'room_assigned_different': room_assigned_different,
    'arrival_month_num': arrival_month_num,
    'lead_time_category': lead_time_category
}])

# PREDICTION
st.divider()

if st.button("🔍 Predict Cancellation Risk"):

    try:
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cancellation Probability", f"{probability:.2%}")
        with col2:
            st.metric("No Cancellation Probability", f"{1 - probability:.2%}")

        st.divider()

        if prediction == 1:
            st.error("🔴 High Risk: Booking likely to be cancelled.")
        else:
            st.success("🟢 Low Risk: Booking likely to be honored.")

    except Exception as e:
        st.error(f"Prediction error: {e}")




