import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from collections import deque
import dl

# Set up the page config
st.set_page_config(page_title="Dashboard UI", layout="wide")

# Initialize the session state for navigation
if "show_graph" not in st.session_state:
    st.session_state["show_graph"] = False

# Page title
st.title("Real-Time Data Streaming Dashboard for Predictive Maintenance of Motors")

# Display KPIs and Health Status Gauge only if "Show Graph" button is not clicked
if not st.session_state["show_graph"]:
    # KPIs Section
    st.markdown("INSTRUMENT 1 INDICATORS")

    # Create columns for KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Fill each KPI with sample values
    with kpi1:
        st.metric(label="Vibration Cluster Deviation", value="1259")
    with kpi2:
        st.metric(label="Vibrational Pattern Error", value="23")
    with kpi3:
        st.metric(label="Require Maintenance", value='YES/NO')

    # Gauge Chart (Tasks)
    st.markdown("## Motor Health Status")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=60,
        title={'text': "Task Completion"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 100], 'color': "lightgreen"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Show Graph button
    if st.button("Show Graph"):
        st.session_state["show_graph"] = True

# Display graphs if "Show Graph" button is clicked
if st.session_state["show_graph"]:
    st.title("Real-Time Data Streaming Dashboard - Graphs")

    # Initialize data buffers for the sliding window (last 20 points)
    window_size = 20
    time_buffer = deque(maxlen=window_size)
    predicted_x, predicted_y, predicted_z = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)
    target_x, target_y, target_z = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)
    mse_buffer = deque(maxlen=window_size)

    # Initial values for the time buffer
    time_buffer.append(0)

    # Placeholders for the graphs
    placeholder_x = st.empty()
    placeholder_y = st.empty()
    placeholder_z = st.empty()
    placeholder_mse = st.empty()

    # Real-time data streaming
    for i in range(200):  # Adjust as needed
        # Append the next time value
        time_buffer.append(time_buffer[-1] + 1)

        # Fetch new data from dl.stream.live() (predicted and target values for each axis and MSE)
        stream_data = dl.stream.live()
        predicted_values = stream_data[1]  # Assume this returns a list or array [x, y, z]
        target_values = stream_data[2]       # Assume this returns a list or array [x, y, z]
        mse_value = stream_data[0]              # MSE directly from the stream data

        # Append new values for x, y, z axes and MSE
        predicted_x.append(predicted_values[0][0])
        predicted_y.append(predicted_values[0][1])
        predicted_z.append(predicted_values[0][2])
        target_x.append(target_values[0][0])
        target_y.append(target_values[0][1])
        target_z.append(target_values[0][2])
        mse_buffer.append(mse_value)

        # Define the plot for x-axis
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=list(time_buffer), y=list(predicted_x), mode="lines+markers", name="Predicted X"))
        fig_x.add_trace(go.Scatter(x=list(time_buffer), y=list(target_x), mode="lines+markers", name="Target X"))
        fig_x.update_layout(title="X-Axis Vibration", xaxis=dict(range=[max(time_buffer) - 5, max(time_buffer)]), yaxis=dict(range=[-1, 1]))

        # Define the plot for y-axis
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(x=list(time_buffer), y=list(predicted_y), mode="lines+markers", name="Predicted Y"))
        fig_y.add_trace(go.Scatter(x=list(time_buffer), y=list(target_y), mode="lines+markers", name="Target Y"))
        fig_y.update_layout(title="Y-Axis Vibration", xaxis=dict(range=[max(time_buffer) - 5, max(time_buffer)]), yaxis=dict(range=[-1, 1]))

        # Define the plot for z-axis
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=list(time_buffer), y=list(predicted_z), mode="lines+markers", name="Predicted Z"))
        fig_z.add_trace(go.Scatter(x=list(time_buffer), y=list(target_z), mode="lines+markers", name="Target Z"))
        fig_z.update_layout(title="Z-Axis Vibration", xaxis=dict(range=[max(time_buffer) - 5, max(time_buffer)]), yaxis=dict(range=[-1, 1]))

        # Define the MSE plot
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(x=list(time_buffer), y=list(mse_buffer), mode="lines+markers", name="MSE"))
        fig_mse.update_layout(title="Mean Square Error (MSE)", xaxis=dict(range=[max(time_buffer) - 5, max(time_buffer)]), yaxis=dict(range=[0, 1]))

        # Display each updated graph
        placeholder_x.plotly_chart(fig_x, use_container_width=True)
        placeholder_y.plotly_chart(fig_y, use_container_width=True)
        placeholder_z.plotly_chart(fig_z, use_container_width=True)
        placeholder_mse.plotly_chart(fig_mse, use_container_width=True)

        # Pause before the next update
        time.sleep(1)
**