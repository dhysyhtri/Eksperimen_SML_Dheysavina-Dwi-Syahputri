import os
import pandas as pd

def preprocess_amazon_sales(input_path, output_path):
    # load dataset
    df = pd.read_csv(input_path)

    # drop columns
    df = df.drop(columns=[
        'index', 'Unnamed: 22',
        'Order ID', 'SKU', 'ASIN',
        'ship-city', 'promotion-ids',
        'fulfilled-by', 'ship-postal-code',
        'ship-country', 'Style'
    ])

    # label engineering
    df['label_profit'] = (
        (df['Amount'] > 0) &
        (df['Status'] != 'Cancelled')
    ).astype(int)

    # feature engineering date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df = df.drop(columns=['Date'])

    # handle missing values
    df['Amount'] = df['Amount'].fillna(0)
    df['Courier Status'] = df['Courier Status'].fillna('Unknown')

    # encoding
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai")
    print("Final shape:", df.shape)
    print("Label distribution:")
    print(df['label_profit'].value_counts())


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    INPUT_PATH = os.path.join(
        BASE_DIR,
        "AmazonSaleReport_raw",
        "Amazon Sale Report.csv"
    )

    OUTPUT_PATH = os.path.join(
        BASE_DIR,
        "Preprocessing",
        "AmazonSaleReport_preprocessing",
        "amazon_sales_clean.csv"
    )

    preprocess_amazon_sales(INPUT_PATH, OUTPUT_PATH)
