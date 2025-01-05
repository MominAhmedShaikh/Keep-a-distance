from datetime import datetime, timedelta
import numpy as np
import yaml
import pandas as pd
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()
URI = os.getenv("URI")



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

KEEPA_ST_ORDINAL = config["keepa_details"]["KEEPA_ST_ORDINAL"]



def get_value(product, key):
    """Helper function to safely extract a key's value from the product."""
    try:
        return product.get('products', [{}])[0].get(key, None)
    except (IndexError, AttributeError):
        return None


def keepa_minutes_to_time(minutes, KEEPA_ST_ORDINAL, to_datetime=True, to_ist=False):
    # Convert minutes to timedelta64[m]
    dt = np.array(minutes, dtype="timedelta64[m]")
    
    # Convert KEEPA_ST_ORDINAL to a datetime64 object if it is a string
    if isinstance(KEEPA_ST_ORDINAL, str):
        KEEPA_ST_ORDINAL = np.datetime64(KEEPA_ST_ORDINAL)
    
    # Shift from ordinal
    dt = KEEPA_ST_ORDINAL + dt

    # Convert to datetime if requested
    if to_datetime:
        dt = dt.astype('datetime64[s]').astype(datetime)
        if to_ist:
            dt = dt + timedelta(hours=5, minutes=30)  # Convert to IST
    return dt

def map_availability(availability):
    mapping = {
        -1: "No Amazon offer exists",
        0: "Amazon offer is in stock and shippable",
        1: "Amazon offer is currently not in stock, but will be in the future (pre-order)",
        2: "Amazon offer availability is unknown",
        3: "Amazon offer is currently not in stock, but will be in the future (back-order)",
        4: "Amazon offer shipping is delayed - see availabilityAmazonDelay for more details"
    }
    return mapping.get(availability, "Unknown status")  # Default value if not found

def process_buybox_data(buyBoxSellerIdHistory, keepa_minutes_to_time):
    processed_seller_info = []

    for i, value in enumerate(buyBoxSellerIdHistory):
        if i % 2 == 0:  # Convert timestamp to datetime
            value = int(value)
            processed_seller_info.append(keepa_minutes_to_time(value,KEEPA_ST_ORDINAL))
        else:
            processed_seller_info.append(value)

    # Split processed_seller_info into two lists: BuyBoxFetchedTime and BuyBoxSeller
    buybox_fetched_time = processed_seller_info[0::2]
    buybox_seller = [
        "BuyBox Suppressed" if seller == -1 else "OOS or No Seller" if seller == -2 else seller
        for seller in processed_seller_info[1::2]
    ]

    # Create and return a DataFrame
    return pd.DataFrame({
        "BuyBoxFetchedTime": buybox_fetched_time,
        "BuyBoxSeller": buybox_seller
    })

def calculate_occupancy_metrics(df, days):
    now = datetime.now()

    # Define the time window
    time_window_start = now - timedelta(days=days)
    time_window_end = now

    # Filter data within the time window
    filtered_df = df[(df["BuyBoxFetchedTime"] >= time_window_start) &
                     (df["BuyBoxFetchedTime"] < time_window_end)].copy()

    if filtered_df.empty:
        return pd.DataFrame(), 0  # Return empty if no data in the window

    # Sort by BuyBoxFetchedTime
    filtered_df = filtered_df.sort_values("BuyBoxFetchedTime")

    # Calculate time difference between consecutive entries
    filtered_df["TimeDiffMinutes"] = filtered_df["BuyBoxFetchedTime"].diff().dt.total_seconds() / 60

    # Create separate dataframes for each case
    df_suppressed = filtered_df[filtered_df["BuyBoxSeller"] == "-1"]
    df_oos_or_no_seller = filtered_df[filtered_df["BuyBoxSeller"] == "-2"]

    # Assign the time difference to the previous seller, ignore suppressed and out-of-stock/no-seller rows
    filtered_df = filtered_df[(filtered_df["BuyBoxSeller"] != "-1") & (filtered_df["BuyBoxSeller"] != "-2")]
    filtered_df["SellerTimeOccupancy"] = filtered_df["TimeDiffMinutes"].shift(-1)

    # Aggregate the total time occupied by each seller (excluding suppressed and out-of-stock/no-seller)
    seller_occupancy = (
        filtered_df.groupby("BuyBoxSeller")["SellerTimeOccupancy"]
        .sum()
        .fillna(0)
        .reset_index()
        .rename(columns={"SellerTimeOccupancy": "TotalTimeOccupiedMinutes"})
    )

    # Aggregate the total time for suppressed and out-of-stock/no-seller
    total_suppressed_minutes = df_suppressed["TimeDiffMinutes"].sum()
    total_oos_or_no_seller_minutes = df_oos_or_no_seller["TimeDiffMinutes"].sum()

    # Calculate the total minutes difference in the filtered dataset
    min_date = filtered_df["BuyBoxFetchedTime"].min()
    max_date = filtered_df["BuyBoxFetchedTime"].max()
    total_minutes_difference = (max_date - min_date).total_seconds() / 60

    # Add occupancy percentage for each seller
    seller_occupancy[f"OccupancyBySellerPercentage{days}D"] = (
        seller_occupancy["TotalTimeOccupiedMinutes"] / total_minutes_difference * 100
    )

    # Count of Suppressed BB Offers
    SuppressedCount = len(df_suppressed)

    # Total Suppression Time
    TotalSuppressionTime = df_suppressed['TimeDiffMinutes'].sum()

    # Suppression percentage
    suppression_percentage = round((TotalSuppressionTime / total_minutes_difference) * 100, 2)  # Suppression time percentage

    # Identify the seller before the first BuyBox suppression
    last_seller_before_suppressed = None
    if not df_suppressed.empty:
        # Find the index of the first suppression event
        latest_suppressed_index = df_suppressed.index.max()

        # Check if there is a previous row
        if latest_suppressed_index > 0:
            # Get the seller before the suppression
            last_seller_before_suppressed = filtered_df.loc[latest_suppressed_index - 1, "BuyBoxSeller"]

    # Create a dictionary to hold the results
    results = {
        "seller_occupancy": seller_occupancy,
        "total_minutes_difference": total_minutes_difference,
        "BuyBoxSuppressed": total_suppressed_minutes,
        "BuyBoxOOSorNoSeller": total_oos_or_no_seller_minutes,
        "SuppressionCount": SuppressedCount,
        "SuppressionPercentage": suppression_percentage,
        "LastSellerBeforeSuppressed": last_seller_before_suppressed,
        "filtered_df": filtered_df,
        "df_suppressed": df_suppressed
    }

    return results

def get_seller_name(api_key, domain, seller_id):

    url = f"https://api.keepa.com/seller?key={api_key}&domain={domain}&seller={seller_id}"
    try:
        # Make the GET request
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        # Parse the JSON response
        seller_info = response.json()

        # Extract the seller name
        seller_name = seller_info.get('sellers', {}).get(seller_id, {}).get('sellerName')
        return seller_name
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return None

def calculate_metrics(sell_price, referral_fee, fba_fees, storage_fee, shipping_to_amazon, cost_price):
    # Calculate Seller Proceeds
    seller_proceeds = sell_price - (referral_fee + fba_fees + storage_fee + shipping_to_amazon)

    # Calculate Net Profit
    net_profit = seller_proceeds - cost_price

    # Calculate Net Margin %
    net_margin_percentage = (net_profit / sell_price) * 100

    # Calculate ROI %
    roi_percentage = (net_profit / cost_price) * 100

    # Return all metrics
    return {
        "Seller Proceeds": round(seller_proceeds, 2),
        "Net Profit": round(net_profit, 2),
        "Net Margin %": round(net_margin_percentage, 2),
        "ROI %": round(roi_percentage, 2)
    }


def connect_to_mongodb():
    """Connects to MongoDB and verifies the connection."""
    client = MongoClient(URI, server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
    except Exception as e:
        print(e)
    return client  # Return the client object

def insert_to_mongodb(client, table_data):

    # Access or create the 'inventory' database
    db = client["Inventory"]

    # Create the 'ProductsMaster' collection
    products_master = db["Products Analysis Master"]

    try:
        result = products_master.insert_one(table_data)
    except Exception as e:
        print(f"Failed to insert data for item : {e}")