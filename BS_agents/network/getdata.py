"""
Load the drivers trajectory data, and process the data files
Determine the GPS locations and time stamps of each record
Save the processed data files as csv

"""

import pandas as pd
from xpinyin import Pinyin
import os
from glob import glob
from datetime import datetime
from chardet import detect


data_directory = 'data'
result_directory = 'results'



# Function to detect the encoding of the CSV file
def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
        encode = detect(raw_data)['encoding']
    print(f"Detected encoding of file: {encode}")
    return encode


def translate_to_pinyin(chinese_string):
    pinyin = Pinyin()
    return pinyin.get_pinyin(chinese_string, tone_marks='marks')


# Function to process each CSV file
def process_file(file_path):
    print(f"Start processing data file: {file_path}")

    # Detect the encoding of the file
    encoding = detect_encoding(file_path)

    # Read the CSV file using pandas with the detected encoding
    df = pd.read_csv(file_path, encoding=encoding)

    """
    # Translate the "城市" and "区县" columns to Pinyin
    print("Translating the city and district data in the file ...")
    df['城市'] = 'Nanjing' #df['城市'].apply(translate_to_pinyin)
    df['区县'] = df['区县'].apply(translate_to_pinyin)
    """

    # Columns to select
    columns_to_keep = ['车辆ID', '状态', '经度', '纬度', '行政区号', '区县', '城市', '数据发送时间']
    df = df[columns_to_keep]

    # Extract date, hour, and minute from the "数据发送时间" column
    print("Extracting the day, hour, and minute in the file ...")
    df['数据发送时间'] = pd.to_datetime(df['数据发送时间'], errors='coerce')
    df['发送日期'] = df['数据发送时间'].dt.date
    df['发送小时'] = df['数据发送时间'].dt.hour
    df['发送分钟'] = df['数据发送时间'].dt.minute

    # Save the processed data into a separate CSV file
    output_file_path = os.path.join(result_directory, f"{os.path.splitext(os.path.basename(file_path))[0]}_data.csv")
    try:
        df.to_csv(output_file_path, index=False, encoding=encoding)
        print(f"The processed data file saved: {output_file_path}")
    except Exception as e:
        print(f"Error saving the file {output_file_path}: {e}")


# Function to process multiple CSV files from a directory
def anl_files():
    # Record the start time
    start_time = datetime.now()

    # Create the output directory if it doesn't exist
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Process each Excel file in the input directory
    for file in os.listdir(data_directory):
        if file.endswith('.csv'):  # Process only .xlsx files
            file_path = os.path.join(data_directory, file)

            # Process the file
            process_file(file_path)

    # Calculate the elapsed time
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()/60
    print(f"Trajectory data analysis completed in {elapsed_time:.4f} minutes.")

