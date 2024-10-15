import streamlit as st
from streamlit_drawable_canvas import st_canvas

import cv2
import requests
import urllib
import json
import os 
# Configs
MODEL_INPUT_SIZE = 28
CANVAS_SIZE = MODEL_INPUT_SIZE * 8

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = os.environ.get("BACKEND_URL")
else:
    BACKEND_URL = "http://localhost:8000"

MODELS_URL = urllib.parse.urljoin(BACKEND_URL, "models")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
EVALUATE_URL = urllib.parse.urljoin(BACKEND_URL, "evaluate")
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")
DELETE_URL = urllib.parse.urljoin(BACKEND_URL, "delete")


st.title("Mnist training and prediction")
st.sidebar.subheader("Page navigtion")
page = st.sidebar.selectbox(label="Select mode", options=[
    "Train", "Evaluate", "Predict", "Delete"])
st.sidebar.write("https://github.com/zademn")

if page == "Train":
    # Conv is not provided yet
    st.session_state.model_type = st.selectbox(
        "Model type", options=["Linear", "Conv"])

    if st.session_state.model_type == "Linear":
        model_name = st.text_input(label="Model name", value="Linear")

        num_layers = st.select_slider(
            label="Number of hidden layers", options=[1, 2, 3])
        cols = st.columns(num_layers)
        hidden_dims = [64] * num_layers
        for i in range(num_layers):
            hidden_dims[i] = cols[i].number_input(
                label=f"Number of neurons in layer {i}", min_value=2, max_value=128, value=hidden_dims[i])

        hyperparams = {
            "input_dim": 28 * 28,
            "hidden_dims": hidden_dims,
            "output_dim": 10,
        }

    if st.session_state.model_type == "Conv":
        model_name = st.text_input(label="Model name", value="Conv")

        num_layers = st.select_slider(
            label="Number of Conv layers", options=[2, 3, 4])
        
        num_filters = st.select_slider(
            label="Size of filters", options=[[16, 32], [32, 64], [64, 128]])

        kernel_size = st.select_slider(
            label="Size of kernel", options=[3, 5, 7])

        stride = st.select_slider(
            label="Number of stride", options=[1, 2, 3])

        dropout_rate = st.select_slider(
            label="Rate of dropout", options=[0, 0.3, 0.5])

        hyperparams = {
            "num_layers": num_layers,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "stride": stride,
            "dropout_rate": dropout_rate,
        }

    epochs = st.number_input("Epochs", min_value=1, value=5, max_value=128)

    if st.button("Train"):
        st.write(f"{hyperparams=}")
        to_post = {"model_name": model_name,
                   "hyperparams": hyperparams, "epochs": epochs,
                   "model_type": st.session_state.model_type}
        response = requests.post(url=TRAIN_URL, data=json.dumps(to_post))
        if response.ok:
            res = response.json()["result"]
        else:
            res = "Training task failed"
        st.write(res)

elif page == "Evaluate":
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")
    
    try:
        response_predict = requests.post(url=EVALUATE_URL,
                                            data=json.dumps({"model_name": model_name}))
        if response_predict.ok:
            res = response_predict.json()
            metrics = res['result']
            # 제목 출력
            st.markdown("### **Evaluation Metrics**:")

            # 먼저 train 관련 항목 출력
            train_metrics = {k: v for k, v in metrics.items() if 'train' in k}
            for metric_name, metric_value in train_metrics.items():
                st.write(f"**{metric_name.replace('_', ' ').capitalize()}**: {metric_value:.4f}")

            # 그다음 나머지 val 관련 항목 출력
            val_metrics = {k: v for k, v in metrics.items() if 'val' in k}
            for metric_name, metric_value in val_metrics.items():
                st.write(f"**{metric_name.replace('_', ' ').capitalize()}**: {metric_value:.4f}")
        else:
            st.write("Some error occured")
    except ConnectionError as e:
        st.write("Couldn't reach backend")


elif page == "Predict":

    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")
    # Setup canvas
    st.write("Draw something here")
    canvas_res = st_canvas(
        fill_color="black",  # Black
        stroke_width=20,
        stroke_color="white",  # White
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key='canvas',
        display_toolbar=True
    )

    # Get image
    if canvas_res.image_data is not None:
        # Scale down image to the model input size
        img = cv2.resize(canvas_res.image_data.astype("uint8"),
                         (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        # Rescaled image upwards to show
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rescaled = cv2.resize(
            img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model input")
        st.image(img_rescaled)

    # Predict on the press of a button
    if st.button("Predict"):
        try:
            response_predict = requests.post(url=PREDICT_URL,
                                             data=json.dumps({"input_image": img.tolist(), "model_name": model_name}))
            if response_predict.ok:
                res = response_predict.json()
                st.markdown(f"**Prediction**: {res['result']}")
                # 클래스 레이블
                class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                # 확률과 클래스 레이블을 묶어 리스트 생성
                softmax_output = res['prob']
                class_probabilities = [(class_labels[i], softmax_output[0][i]) for i in range(len(class_labels))]

                # 확률이 큰 순서대로 정렬
                sorted_probabilities = sorted(class_probabilities, key=lambda x: x[1], reverse=True)

                # 테이블 형식으로 변환
                table_data = [["Class", "Probability"]]  # 열 이름 추가
                for class_label, probability in sorted_probabilities:
                    table_data.append([class_label, f"{probability * 100:.4f}%"])  # 확률을 %로 포맷팅하여 추가

                # 제목 출력
                st.markdown("### **Softmax Output Probabilities (Sorted)**")

                # 클래스별 확률을 테이블 형식으로 출력
                st.table(table_data)
                
            else:
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")

elif page == "Delete":
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")

    to_post = {"model_name": model_name}
    # Delete on the press of a button
    if st.button("Delete"):
        try:
            response = requests.post(url=DELETE_URL,
                                     data=json.dumps(to_post))
            if response.ok:
                res = response.json()
                st.write(res["result"])
            else:
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")
else:
    st.write("Page does not exist")
