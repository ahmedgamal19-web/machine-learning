import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.title("Linear Regression Model")

# Initialize session state to hold the data and model
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None


def apply_encoding(data, columns_encoding):
    for column, encoding_type in columns_encoding.items():
        if encoding_type == "Label Encoding":
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
        elif encoding_type == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=[column])
        elif encoding_type == "Ordinal Encoding":
            encoder = OrdinalEncoder()
            data[column] = encoder.fit_transform(data[[column]])
    return data

def apply_scaling(data, columns_scaling):
    for column, scaling_type in columns_scaling.items():
        if scaling_type == "Standard Scaling":
            scaler = StandardScaler()
            data[[column]] = scaler.fit_transform(data[[column]])
        elif scaling_type == "Min-Max Scaling":
            scaler = MinMaxScaler()
            data[[column]] = scaler.fit_transform(data[[column]])
    return data

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload and Preview Data", "Preprocessing", "Build Model", "Model Evaluation", "Real-time Predictions"])

# Tab 1: Upload and Preview Data
with tab1:
    st.header("Upload and Preview Data")
    # Image path
    st.image(r"C:\Users\King\Downloads\Machine-Learning-Services-banner.png", caption="Overview of Data Analysis", use_column_width=True)


    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(st.session_state.data.head())

        st.write("### Data Info:")
        st.write(st.session_state.data.info())

        # Correlation heatmap
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            st.write("### Correlation Heatmap:")
            correlation_matrix = st.session_state.data[numeric_cols].corr()
            fig = ff.create_annotated_heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.index.tolist(),
                annotation_text=correlation_matrix.round(2).astype(str).values,
                colorscale='Viridis'
            )
            fig.update_layout(title='Correlation Heatmap', width=700, height=500)
            st.plotly_chart(fig)
        else:
            st.write("No numeric columns available for correlation.")
        
    else:
        st.write("Please upload a CSV file to start.")

# Tab 2: Preprocessing
with tab2:
    st.header("Data Preprocessing")
    
    if st.session_state.data is not None:
        # Encoding options for each column
        encoding_columns = {}
        for column in st.session_state.data.columns:
            if st.session_state.data[column].dtype == 'object':
                encoding_type = st.selectbox(f"Choose encoding for {column}", ["None", "Label Encoding", "One-Hot Encoding", "Ordinal Encoding"], key=f"encode_{column}")
                if encoding_type != "None":
                    encoding_columns[column] = encoding_type
        
        # Scaling options for each column
        scaling_columns = {}
        for column in st.session_state.data.columns:
            if st.session_state.data[column].dtype != 'object':
                scaling_type = st.selectbox(f"Choose scaling for {column}", ["None", "Standard Scaling", "Min-Max Scaling"], key=f"scale_{column}")
                if scaling_type != "None":
                    scaling_columns[column] = scaling_type

        # Button to apply preprocessing
        if st.button("Apply Preprocessing"):
            st.session_state.data = apply_encoding(st.session_state.data, encoding_columns)
            st.session_state.data = apply_scaling(st.session_state.data, scaling_columns)
            st.write("### Data after Preprocessing:")
            st.dataframe(st.session_state.data)

            # Button to download the preprocessed data
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download Preprocessed Data as CSV",
                data=csv,
                file_name='preprocessed_data.csv',
                mime='text/csv'
            )
    else:
        st.write("Please upload data in Tab 1 to preprocess.")

# Tab 3: Build Model


with tab3:
    st.header("Build Model")

    if st.session_state.data is not None:
        st.session_state.target_column = st.selectbox("Select Target Column", options=st.session_state.data.columns)
        st.session_state.feature_columns = st.multiselect("Select Feature Columns", options=[col for col in st.session_state.data.columns if col != st.session_state.target_column])

        # Display correlation of selected feature columns with the target
        if st.session_state.feature_columns:
            correlation_matrix = st.session_state.data.corr()
            target_correlation = correlation_matrix[st.session_state.target_column][st.session_state.feature_columns].sort_values(ascending=False)

            st.write("### Correlation of Selected Features with Target Variable:")
            fig_corr = px.bar(
                x=target_correlation.index,
                y=target_correlation.values,
                text=target_correlation.apply(lambda x: f"{x:.2f}"),
                labels={'x': 'Features', 'y': 'Correlation Coefficient'},
                title='Feature Correlation with Target Variable'
            )
            fig_corr.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_corr)

        if st.button("Build Model"):
            X = st.session_state.data[st.session_state.feature_columns]
            y = st.session_state.data[st.session_state.target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.session_state.model = LinearRegression()
            st.session_state.model.fit(X_train, y_train)
            
            y_pred = st.session_state.model.predict(X_test)
            
            # Store the test and predicted values for the evaluation tab
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            # Compute metrics
            accuracy = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Display metrics in a visually appealing way
            st.write("### Model Evaluation Metrics")

            # Gauge chart for accuracy
            accuracy_percentage = accuracy * 100  # Convert to percentage

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy_percentage,
                title={'text': "Model Accuracy (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'bgcolor': "lightgray",
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': accuracy_percentage,
                    }
                }
            ))

            st.plotly_chart(fig_gauge)

            # Display other metrics
            st.write(f"### Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"### Mean Absolute Error (MAE): {mae:.4f}")
    else:
        st.write("Please preprocess the data in Tab 2 before building the model.")


# Tab 4: Model Evaluation
with tab4:
    st.header("Model Evaluation")
    
    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
        st.write("### Predictions vs Actual (Best Fit Line)")
        eval_data = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.y_pred})
        
        # Create scatter plot
        fig = px.scatter(eval_data, x="Actual", y="Predicted", title="Predictions vs Actual")
        
        # Plot the best fit line manually
        line_fig = px.line(eval_data, x="Actual", y="Actual", title="Best Fit Line")
        
        # Combine scatter plot and best fit line
        fig.add_traces(line_fig.data)
        
        st.plotly_chart(fig)
    else:
        st.write("Please build the model in Tab 3 to evaluate.")

# Tab 5: Real-time Predictions
with tab5:
    st.header("Real-time Predictions")
    
    if st.session_state.model is not None and st.session_state.feature_columns is not None:
        st.write("### Enter Feature Values for Prediction")

        # Create input fields for each feature
        prediction_inputs = {}
        for feature in st.session_state.feature_columns:
            prediction_inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

        if st.button("Predict"):
            input_data = pd.DataFrame([prediction_inputs])
            prediction = st.session_state.model.predict(input_data)[0]
            st.write(f"### Predicted Value for {st.session_state.target_column}: {prediction:.4f}")
    else:
        st.write("Please build the model in Tab 3 before making predictions.")