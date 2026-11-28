from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date

from prediction import predict_booking, MODEL_NAME, THRESHOLD

from preprocessing import preprocess_input
from prediction import load_model

app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description="API dự đoán khả năng hủy đặt phòng",
    version="1.0.0"
)


# Khai báo cấu trúc dữ liệu đầu vào
class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date: date
    reserved_room_type: str
    adr: float
    deposit_type: str
    adults: int
    children: int
    babies: int
    country: str
    customer_type: str
    is_repeated_guest: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    meal: str
    required_car_parking_spaces: int
    total_of_special_requests: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    market_segment: str
    distribution_channel: str
    agent: int | None = None
    company: int | None = None


# Kiểm tra trạng thái API
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "API is running"
    }


# Thông tin mô hình
@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "threshold": THRESHOLD
    }


# Dự đoán một đơn đặt phòng
@app.post("/predict")
def predict(booking: BookingInput):
    try:
        input_data = booking.model_dump(mode="json")

        result = predict_booking(input_data)

        return {
            "model": MODEL_NAME,
            "threshold": THRESHOLD,
            **result.iloc[0].to_dict()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    
@app.post("/predict-detail")
def predict_detail(booking: BookingInput):
    try:
        # Bước 1: Tiếp nhận dữ liệu đầu vào
        input_data = booking.model_dump(mode="json")

        # Bước 2: Feature Engineering
        features = preprocess_input(input_data)

        # Bước 3: Tiền xử lý và mã hóa dữ liệu
        model = load_model()

        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(features)

        # Bước 4: Dự đoán bằng LightGBM
        result = predict_booking(input_data)
        output = result.iloc[0].to_dict()

        # Bước 5: Trả về chi tiết quy trình
        return {
            "step_1_input": {
                "number_of_fields": len(input_data),
                "data": input_data
            },

            "step_2_feature_engineering": {
                "number_of_features": features.shape[1],
                "features": features.iloc[0].to_dict()
            },

            "step_3_preprocessing": {
                "methods": [
                    "Median Imputation",
                    "One-Hot Encoding"
                ],
                "number_of_features_before": features.shape[1],
                "number_of_features_after": transformed.shape[1]
            },

            "step_4_model_inference": {
                "model": MODEL_NAME,
                "cancel_probability": output["cancel_probability"],
                "not_cancel_probability": output["not_cancel_probability"]
            },

            "step_5_decision": {
                "threshold": THRESHOLD,
                "prediction": output["prediction"],
                "prediction_label": output["prediction_label"]
            },

            "step_6_output": output
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




    """
    {
  "hotel": "City Hotel",
  "lead_time": 90,
  "arrival_date": "2017-08-25",
  "reserved_room_type": "A",
  "adr": 120,
  "deposit_type": "No Deposit",
  "adults": 2,
  "children": 1,
  "babies": 0,
  "country": "PRT",
  "customer_type": "Transient",
  "is_repeated_guest": 0,
  "stays_in_weekend_nights": 2,
  "stays_in_week_nights": 3,
  "meal": "BB",
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 1,
  "previous_cancellations": 2,
  "previous_bookings_not_canceled": 8,
  "market_segment": "Online TA",
  "distribution_channel": "TA/TO",
  "agent": 9,
  "company": null
}
    
    """