from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import ta


# Settings
width_px = 1000
#ta_col_prefix = 'ta_'


# Sidebar
return_value = st.sidebar.selectbox(
    'How many periods to calculate the Mean Predicted Diesel Quality?',
     [1, 2, 3, 5, 7, 14, 31])


# Data preparation
@st.cache(allow_output_mutation=True)
def load_data():

    # Load dataset
    df = pd.read_csv('/Users/rraman7/python_local/streamlit_dr_app/data.csv', sep=',') #SET PATH HERE

    # Clean NaN values
    df = ta.utils.dropna(df)

    # Apply feature engineering
    # 

    return df

df = load_data()

# Prepare target: X Periods Return
df['Diesel Quality-Mean Predicted Over a Batch'] = (df['Diesel Quality-DataRobot Prediction'] / df['Diesel Quality-DataRobot Prediction'].shift(return_value) - 1) * 100

# Clean NaN values
df = df.dropna()

# Body
st.title('Diesel Quality Predictions and Insights')

a = datetime.utcfromtimestamp(df['Timestamp'].head(1)).strftime('%Y-%m-%d %H:%M:%S')
b = datetime.utcfromtimestamp(df['Timestamp'].tail(1)).strftime('%Y-%m-%d %H:%M:%S')
st.write(f'DataRobot Diesel Quality Model Predictions and Insights')
st.write('The dashboard shows the predictions from the DataRobot Model along with best'
		 'performing features in the model')

st.subheader('Dataframe')
st.write(df)

st.subheader('Describe dataframe')
st.write(df.describe())

st.write('Number of rows: {}, Number of columns: {}'.format(*df.shape))

st.subheader('Diesel Quality: Predicted vs Actual')
st.line_chart(df[['Diesel Quality-DataRobot Prediction','Diesel Quality-Actual Value']], width=width_px)

st.subheader(f'Change in Predicted Diesel Quality Over {return_value} periods')
st.area_chart(df['Diesel Quality-Mean Predicted Over a Batch'], width=width_px)

st.subheader('Predicted Diesel Quality Histogram')
bins = list(np.arange(-10, 10, 0.5))
hist_values, hist_indexes = np.histogram(df['Diesel Quality-Mean Predicted Over a Batch'], bins=bins)
st.bar_chart(pd.DataFrame(data=hist_values, index=hist_indexes[0:-1]), width=width_px)
st.write('Predicted Diesel Quality value min: {0:.2f}%; max: {1:.2f}%; mean: {2:.2f}%; std: {3:.2f}'.format(
    np.min(df['Diesel Quality-Mean Predicted Over a Batch']), np.max(df['Diesel Quality-Mean Predicted Over a Batch']), np.mean(df['Diesel Quality-Mean Predicted Over a Batch']), np.std(df['Diesel Quality-Mean Predicted Over a Batch'])))

# Univariate Analysis
st.subheader('Correlation coefficient top features and Mean Predicted Diesel Quality')

x_cols = [col for col in df.columns if col not in ['Timestamp', 'Diesel Quality-Mean Predicted Over a Batch',"Diesel Quality-DataRobot Prediction","Diesel Quality-Actual Value"]]
labels = [col for col in x_cols]
values = [np.corrcoef(df[col], df['Diesel Quality-Mean Predicted Over a Batch'])[0, 1] for col in x_cols]

st.bar_chart(data=pd.DataFrame(data=values, index=labels), width=width_px)
