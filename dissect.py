from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bs4 import BeautifulSoup
from functools import reduce
from bson import json_util
# from dotenv import load_dotenv
from sp_api.api import Products
import numpy as np
import math
import pandas as pd
import warnings
import re
import requests
import json
import yaml
import os
from utils import *

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Enter Code")

# load_dotenv()

KEEPA_API_KEY                = os.environ["KEEPA_API_KEY"]
LWA_APP_ID                   = os.environ["LWA_APP_ID"]
LWA_CLIENT_SECRET            = os.environ["LWA_CLIENT_SECRET"]
SP_API_REFRESH_TOKEN         = os.environ["SP_API_REFRESH_TOKEN"]
DATABASE_NAME                = os.environ["DATABASE_NAME"]
ERROR_COLLECTION             = os.environ["ERROR_COLLECTION"]
PRODUCT_ANALYSIS_COLLECTION  = os.environ["PRODUCT_ANALYSIS_COLLECTION"]
PRODUCT_COLLECTION           = os.environ["PRODUCT_COLLECTION"]
PRODUCT_INVENTORY_COLLECTION = os.environ["PRODUCT_INVENTORY_COLLECTION"]
URI                          = os.environ["URI"]

try:
    KEEPA_API_KEY = os.environ["KEEPA_API_KEY"]
except KeyError:
    raise RuntimeError("Missing required environment variable: KEEPA_API_KEY")

client = MongoClient(URI)
db = client[DATABASE_NAME]
error_collection = db[ERROR_COLLECTION]
product_analysis_collection = db[PRODUCT_ANALYSIS_COLLECTION]
product_collection = db[PRODUCT_COLLECTION]
product_inventory_collection = db[PRODUCT_INVENTORY_COLLECTION]

documents = product_collection.find()

print("Enter Code 1")


# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)

KEEPA_ST_ORDINAL = "2011-01-01"
only_live_offers = 1
domain           = 1
update           = 1
history          = 1
rating           = 1
buybox           = 1
offers_k         = 100
stock            = 1


print("Enter Code 2")


for document_num,document in enumerate(documents):
    asin = document.get("ASIN", "ASIN not found")
    vendor_sku = document.get("Vendor SKU", "Vendor SKU not found")
    print(document_num,asin,vendor_sku)
    try:
        url = f"https://api.keepa.com/product?key={KEEPA_API_KEY}&domain={domain}&asin={asin}&update={update}&history={history}&only-live-offers={only_live_offers}&rating={rating}&buybox={buybox}&stock={stock}&offers={offers_k}"
        print(url)
        payload = {}
        headers = {}

        product = requests.request("GET", url, headers=headers, data=payload)
        product = json.loads(product.text)
        print(product)

        # print(f"Product structure: {product}")  # Debug: Print the entire product dictionary

        tokens_left = product.get("tokensLeft")
        # print(f"Tokens left: {tokens_left}")  # Debug: Check the value of tokens_left

        while tokens_left <= 0:
            refill_in = response.get("refillIn", 5000) / 1000.0  # Convert milliseconds to seconds
            print(f"Tokens depleted. Waiting {refill_in} seconds for refill...")
            time.sleep(refill_in)
            tokens_left = product.get("tokensLeft")  # Refresh tokens_left after waiting
            print(f"Tokens left (after waiting): {tokens_left}")  # Debug: Check updated tokens_left

        # Handle error key
        if "error" in product:
            error = product["error"]
            error_data = {
                "asin": asin,
                "error_type": error.get("type"),
                "error_message": error.get("message"),
                "details": error.get("details"),
                "timestamp": product.get("timestamp"),
            }
            error_collection.insert_one(error_data)
            print(f"Inserted error for ASIN {asin} into MongoDB.")

        elif not product['products'][0].get('liveOffersOrder'):
            print(f'Skipping this {asin} as it has no offers or is untraceable by KEEPA')

        # Handle product key
        elif "products" in product:
            
            fields = [
                # ASIN Details Update Info
                'lastUpdate', 'lastPriceChange', 'trackingSince', 'lastRatingUpdate', 
                'listedSince', 'lastStockUpdate', 'lastSoldUpdate',

                # ASIN Details
                'manufacturer', 'title', 'rootCategory', 'asin', 'productGroup', 
                'color', 'size', 'categories', 'categoryTree', 'upcList',

                # ASIN Details Flags
                'isAdultProduct', 'newPriceIsMAP', 'availabilityAmazon', 'isB2B',

                # ASIN Offer Details
                'liveOffersOrder', 'monthlySold', 'offers', 'buyBoxSellerIdHistory', 
                'salesRanks', 'reviews', 'monthlySoldHistory',

                # ASIN Manufacturer Details
                'partNumber', 'brand',

                # ASIN Shipping and Item Dimensions
                'packageHeight', 'packageLength', 'packageWidth', 'packageWeight', 
                'packageQuantity', 'itemHeight', 'itemLength', 'itemWidth', 'itemWeight',

                # ASIN Fees
                'fbaFees', 'referralFeePercent', 'referralFeePercentage'
            ]

            # Populate fields dynamically and unpack directly
            populated_fields = {key: get_value(product, key) for key in fields}

            # Unpack variables directly
            (lastUpdate, lastPriceChange, trackingSince, lastRatingUpdate, listedSince, lastStockUpdate, lastSoldUpdate, manufacturer, title, rootCategory, asin, 
            productGroup, color, size, categories, categoryTree, upcList, isAdultProduct,newPriceIsMAP, availabilityAmazon, isB2B, liveOffersOrder, monthlySold, offers, 
            buyBoxSellerIdHistory, salesRanks, reviews, monthlySoldHistory, partNumber, brand, packageHeight, packageLength, packageWidth, packageWeight, packageQuantity, 
            itemHeight, itemLength, itemWidth, itemWeight, fbaFees, referralFeePercent, referralFeePercentage) = (populated_fields[key] for key in fields)


            product_df = pd.DataFrame()

            product_df['listing.asin']                        = [populated_fields['asin']]
            product_df['listing.listedSince']                 = [keepa_minutes_to_time(populated_fields['listedSince'],KEEPA_ST_ORDINAL)]
            product_df['tracking.lastMonthlySoldUpdate']      = [keepa_minutes_to_time(populated_fields['lastSoldUpdate'],KEEPA_ST_ORDINAL)]
            product_df['stock.lastStockUpdate']               = [keepa_minutes_to_time(populated_fields['lastStockUpdate'],KEEPA_ST_ORDINAL)]
            product_df['listing.manufacturer']                = [populated_fields['manufacturer']]
            product_df['listing.title']                       = [populated_fields['title']]
            product_df['listing.rootCategory']                = [populated_fields['rootCategory']]
            product_df['listing.productGroup']                = [populated_fields['productGroup']]
            product_df['listing.color']                       = [populated_fields['color']]
            product_df['listing.size']                        = [populated_fields['size']]
            product_df['listing.upcList']                     = ["".join(i for i in populated_fields['upcList']) if isinstance(populated_fields['upcList'], list) else None]
            product_df['listing.isAdultProduct']              = [populated_fields['isAdultProduct']]
            product_df['listing.newPriceIsMAP']               = [populated_fields['newPriceIsMAP']]
            product_df['listing.isB2B']                       = [populated_fields['isB2B']]
            product_df['listing.availabilityAmazon']          = [map_availability(populated_fields['availabilityAmazon'])]
            product_df['analysis.upcCount']                   = [len(populated_fields['upcList']) if isinstance(populated_fields['upcList'], list) else None]
            product_df['analysis.countOfOffers']              = [len(populated_fields['offers']) if isinstance(populated_fields.get('offers'), (list, tuple)) else None]
            product_df['analysis.monthlySold']                = [populated_fields['monthlySold']]
            product_df['listing.listedSince']                 = product_df['listing.listedSince'].astype('str')
            product_df['tracking.lastMonthlySoldUpdate']      = product_df['tracking.lastMonthlySoldUpdate'].astype('str')
            product_df['stock.lastStockUpdate']               = product_df['stock.lastStockUpdate'].astype('str')

            # Initialize an empty list to hold the analysis results
            all_analysis = []

            # Iterate through each offer in 'offers'
            for i in range(len(offers)):
                sellerId = offers[i]['sellerId']
                condition = offers[i]['condition']
                isPrime = offers[i]['isPrime']
                isMAP = offers[i]['isMAP']
                isShippable = offers[i]['isShippable']
                isScam = offers[i]['isScam']
                isAmazon = offers[i]['isAmazon']
                isFBA = offers[i]['isFBA']
                shipsFromChina = offers[i]['shipsFromChina']

                offer_csv = offers[i]['offerCSV']

                # Processed output array for each offer
                preprocessed_array = []
                for j in range(0, len(offer_csv), 3):
                    datetime_value = keepa_minutes_to_time(offer_csv[j],KEEPA_ST_ORDINAL)  # Assuming this function exists
                    price = offer_csv[j + 1] / 100 if offer_csv[j + 1] > 0 else 0
                    shipping_price = offer_csv[j + 2] / 100 if offer_csv[j + 2] > 0 else 0
                    preprocessed_array.extend([datetime_value, price, shipping_price])

                # Reshape the data
                offers_reshaped = [
                    (preprocessed_array[i], preprocessed_array[i + 1] + preprocessed_array[i + 2])
                    for i in range(0, len(preprocessed_array), 3)
                ]

                # Extract required data
                prices = [price for _, price in offers_reshaped]
                timestamps = [time for time, _ in offers_reshaped]

                # Existing Analysis
                lowest_price = min(prices)
                highest_price = max(prices)
                lowest_price_time = timestamps[prices.index(lowest_price)]
                highest_price_time = timestamps[prices.index(highest_price)]

                average_price = round(np.mean(prices), 2)

                earliest_date = min(timestamps)
                latest_date = max(timestamps)
                date_difference = latest_date - earliest_date

                offer_count = len(prices)

                percentage_changes = [
                    round(((prices[i] - prices[i - 1]) / prices[i - 1]) * 100, 2)
                    for i in range(1, len(prices))
                ]
                average_percentage_change = round(np.mean(percentage_changes), 2)

                is_latest_above_mean = prices[-1] > average_price

                # New Analysis
                current_offer = prices[-1]
                previous_offer = prices[-2] if len(prices) > 1 else None  # Check if at least 2 offers exist

                # Calculate percentage difference between current and previous offer
                current_previous_percentage_diff = (
                    round(((current_offer - previous_offer) / previous_offer) * 100, 2)
                    if previous_offer and previous_offer != 0
                    else None
                )

                # Calculate time difference between current and previous offer
                current_previous_time_diff = (
                    timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else None
                )


                # Collect the analysis results
                analysis = {
                    "sellerId": sellerId,
                    "condition": condition,
                    "isPrime": isPrime,
                    "isMAP": isMAP,
                    "isShippable": isShippable,
                    "isScam": isScam,
                    "isAmazon": isAmazon,
                    "isFBA": isFBA,
                    "shipsFromChina": shipsFromChina,
                    "Lowest Price": lowest_price,
                    "Highest Price": highest_price,
                    "Lowest Price Time": lowest_price_time,
                    "Highest Price Time": highest_price_time,
                    "Average Price": average_price,
                    "Earliest Date": earliest_date,
                    "Latest Date": latest_date,
                    "Offer Count": offer_count,
                    "Average Percentage Change": average_percentage_change,
                    "Is Latest Offer Above Mean": is_latest_above_mean,
                    "Current Offer": current_offer,
                    "Previous Offer": previous_offer,
                    "Current vs Previous Time Difference": current_previous_time_diff,
                }

                # Append the analysis to the list
                all_analysis.append(analysis)

            # Create a DataFrame from the analysis results
            offers_df = pd.DataFrame(all_analysis)
            offers_df = offers_df.replace({np.nan: None, pd.NaT: None})

            # Split the offers DataFrame into FBA and FBM sellers
            fba_sellers = offers_df[offers_df['isFBA'] == True]
            fbm_sellers = offers_df[offers_df['isFBA'] == False]

            # Initialize variables for lowest FBA and FBM seller details
            lowest_fba_seller = None
            lowest_fbm_seller = None

            # Find the lowest FBA seller if fba_sellers is not empty
            if not fba_sellers.empty and 'Current Offer' in fba_sellers.columns and not fba_sellers['Current Offer'].isnull().all():
                lowest_fba_seller = fba_sellers.loc[fba_sellers['Current Offer'].idxmin()]

            # Find the lowest FBM seller if fbm_sellers is not empty
            if not fbm_sellers.empty and 'Current Offer' in fbm_sellers.columns and not fbm_sellers['Current Offer'].isnull().all():
                lowest_fbm_seller = fbm_sellers.loc[fbm_sellers['Current Offer'].idxmin()]

            # Update the product DataFrame with seller counts
            product_df['analysis.FBAoffersCount'] = [len(fba_sellers)]
            product_df['analysis.FBMoffersCount'] = [len(fbm_sellers)]

            # Update the product DataFrame with FBA seller details or None
            if lowest_fba_seller is not None:
                for row in lowest_fba_seller.index:
                    product_df[f'analysis.LowestFbaSellerDetail.{row}'] = [lowest_fba_seller[row]]
            else:
                for col in offers_df.columns:
                    product_df[f'analysis.LowestFbaSellerDetail.{col}'] = [None]

            # Update the product DataFrame with FBM seller details or None
            if lowest_fbm_seller is not None:
                for row in lowest_fbm_seller.index:
                    product_df[f'analysis.LowestFbmSellerDetail.{row}'] = [lowest_fbm_seller[row]]
            else:
                for col in offers_df.columns:
                    product_df[f'analysis.LowestFbmSellerDetail.{col}'] = [None]

            # Convert specific columns to string type
            columns_to_convert = [
                'analysis.LowestFbaSellerDetail.Lowest Price Time',
                'analysis.LowestFbaSellerDetail.Highest Price Time',
                'analysis.LowestFbaSellerDetail.Earliest Date',
                'analysis.LowestFbaSellerDetail.Latest Date',
                'analysis.LowestFbaSellerDetail.Current vs Previous Time Difference',
                'analysis.LowestFbmSellerDetail.Lowest Price Time',
                'analysis.LowestFbmSellerDetail.Highest Price Time',
                'analysis.LowestFbmSellerDetail.Earliest Date',
                'analysis.LowestFbmSellerDetail.Latest Date',
                'analysis.LowestFbmSellerDetail.Current vs Previous Time Difference'
            ]

            for col in columns_to_convert:
                if col in product_df.columns:
                    product_df[col] = product_df[col].astype('str')

            processed_seller_info = []

            for i in range(0, len(buyBoxSellerIdHistory)):
                value = buyBoxSellerIdHistory[i]

                if i % 2 == 0:  # Check for every 1st, 3rd, 5th element (index 0, 2, 4)
                    value = int(value)  # Convert the value to int

                # If it's one of the datetime-related elements, apply the datetime conversion
                if i % 2 == 0:  # Convert the integer value to datetime
                    datetime_value = keepa_minutes_to_time(value,KEEPA_ST_ORDINAL)
                    processed_seller_info.append(datetime_value)
                else:
                    processed_seller_info.append(value)


            buybox_fetched_time = []
            buybox_seller = []

            # Iterate over the processed_seller_info array
            for i in range(0, len(processed_seller_info), 2):
                buybox_fetched_time.append(processed_seller_info[i])
                seller = processed_seller_info[i + 1]
                if seller == -1:
                    buybox_seller.append("BuyBox Suppressed")
                elif seller == -2:
                    buybox_seller.append("OOS or No Seller")
                else:
                    buybox_seller.append(seller)

            buybox_df = process_buybox_data(buyBoxSellerIdHistory, keepa_minutes_to_time)
            EarliestDateBuyBoxAppearance = buybox_df["BuyBoxFetchedTime"].min()
            LatestDateBuyBoxAppearance = buybox_df["BuyBoxFetchedTime"].max()

            product_df['analysis.EarliestDateBuyBoxAppearance'] = [buybox_df["BuyBoxFetchedTime"].min()]
            product_df['analysis.LatestDateBuyBoxAppearance']   = [buybox_df["BuyBoxFetchedTime"].max()]
            product_df['analysis.DaysDiffBuyBoxAppearance']     = [(LatestDateBuyBoxAppearance - EarliestDateBuyBoxAppearance).days]
            product_df['analysis.FirstDateBuyBoxAppearance']    = [(EarliestDateBuyBoxAppearance - keepa_minutes_to_time(listedSince,KEEPA_ST_ORDINAL))]

            product_df['analysis.EarliestDateBuyBoxAppearance'] = product_df['analysis.EarliestDateBuyBoxAppearance'].astype('str')
            product_df['analysis.LatestDateBuyBoxAppearance']   = product_df['analysis.LatestDateBuyBoxAppearance'].astype('str')
            product_df['analysis.DaysDiffBuyBoxAppearance']     = product_df['analysis.DaysDiffBuyBoxAppearance'].astype('str')
            product_df['analysis.FirstDateBuyBoxAppearance']    = product_df['analysis.FirstDateBuyBoxAppearance'].astype('str')

            metrics_30d = calculate_occupancy_metrics(buybox_df, 30)
            metrics_60d = calculate_occupancy_metrics(buybox_df, 60)
            metrics_90d = calculate_occupancy_metrics(buybox_df, 90)
            metrics_180d = calculate_occupancy_metrics(buybox_df,180)
            metrics_360d = calculate_occupancy_metrics(buybox_df,360)


            metrics_df = [
                metrics_30d['seller_occupancy'].rename(columns={'TotalTimeOccupiedMinutes':'TotalTimeOccupiedMinutes30D'}),
                metrics_60d['seller_occupancy'].rename(columns={'TotalTimeOccupiedMinutes':'TotalTimeOccupiedMinutes60D'}),
                metrics_90d['seller_occupancy'].rename(columns={'TotalTimeOccupiedMinutes':'TotalTimeOccupiedMinutes90D'}),
                metrics_180d['seller_occupancy'].rename(columns={'TotalTimeOccupiedMinutes':'TotalTimeOccupiedMinutes180D'}),
                metrics_360d['seller_occupancy'].rename(columns={'TotalTimeOccupiedMinutes':'TotalTimeOccupiedMinutes360D'})
            ]

            # Perform an outer join on all DataFrames using `reduce`
            metrics_merged_df = reduce(lambda left, right: pd.merge(left, right, on='BuyBoxSeller', how='outer'), metrics_df)
            metrics_merged_df['SellerName'] = metrics_merged_df['BuyBoxSeller'].apply(
                lambda x: get_seller_name(KEEPA_API_KEY, domain, x)
            )

            metrics_merged_df.fillna(0)
            metrics_merged_df = metrics_merged_df.where(metrics_merged_df.notnull(), None)
            product_df['analysis.BuyBoxHistAnalysis'] = [metrics_merged_df.to_json(orient='records')] * len(product_df)

            all_stock_data = pd.DataFrame()

            for offer in offers:
                if 'stockCSV' in offer.keys():
                    stockData = offer['stockCSV']
                    
                    # Create a DataFrame for the current offer
                    stockDataDf = pd.DataFrame({
                        'SelledId': offer['sellerId'],
                        'StockFetchedDate': stockData[::2],
                        'StockAvailablilityCount': stockData[1::2]
                    })
                    
                    # Apply the keepa_minutes_to_time conversion
                    stockDataDf['StockFetchedDate'] = stockDataDf['StockFetchedDate'].apply(lambda x : keepa_minutes_to_time(x,KEEPA_ST_ORDINAL))
                    
                    # Add the derived columns
                    stockDataDf['OpeningStock'] = stockDataDf['StockAvailablilityCount'].shift(1, fill_value=0)
                    stockDataDf['ClosingStock'] = stockDataDf['StockAvailablilityCount']
                    stockDataDf['BalanceStock'] = stockDataDf['ClosingStock'] - stockDataDf['OpeningStock']
                    stockDataDf['SoldStock']    = stockDataDf['OpeningStock'] - stockDataDf['ClosingStock']
                    stockDataDf['SoldStock']    = stockDataDf['SoldStock'].clip(lower=0)
                    stockDataDf['Year']         = stockDataDf['StockFetchedDate'].dt.year
                    stockDataDf['Month']        = stockDataDf['StockFetchedDate'].dt.month
                    
                    # Append to the cumulative DataFrame
                    all_stock_data = pd.concat([all_stock_data, stockDataDf], ignore_index=True)



            # Group and sort the data
            monthly_avg_stock = all_stock_data.groupby(['SelledId', 'Year', 'Month'])['SoldStock'].sum().reset_index()
            monthly_stock_sold = monthly_avg_stock.sort_values(by=['Year', 'Month'], ascending=[True, True]).reset_index()

            monthly_stock_sold.drop(columns=['index'], inplace=True)

            # Get unique seller IDs
            unique_seller_ids = monthly_stock_sold['SelledId'].unique()

            seller_name_mapping = {seller_id: get_seller_name(KEEPA_API_KEY, domain, seller_id) for seller_id in unique_seller_ids}

            # Map the SellerName back to the original DataFrame
            monthly_stock_sold['SellerName'] = monthly_stock_sold['SelledId'].map(seller_name_mapping)

            monthly_stock_sold = monthly_stock_sold[['SellerName','Month','Year','SoldStock']]
            monthly_stock_sold = monthly_stock_sold[monthly_stock_sold['SoldStock']!=0]
            monthly_stock_sold = monthly_stock_sold.where(monthly_stock_sold.notnull(), None)

            product_df['analysis.StockHistAnalysis'] = [monthly_stock_sold.to_json(orient='records')] * len(product_df)

            # Ensure monthlySoldHistory is not None and has at least some data
            if monthlySoldHistory is not None and len(monthlySoldHistory) >= 2:
                # Create a DataFrame with fetched dates and counts
                MonthlySoldDf = pd.DataFrame({
                    'MonthlySoldFetchedDate': monthlySoldHistory[::2],
                    'MonthlySoldCount': monthlySoldHistory[1::2]
                })
                
                # Convert Keepa minutes to datetime
                MonthlySoldDf['MonthlySoldFetchedDate'] = MonthlySoldDf['MonthlySoldFetchedDate'].apply(
                    lambda x: keepa_minutes_to_time(x, KEEPA_ST_ORDINAL)
                )
                
                # Extract Year and Month
                MonthlySoldDf['Year'] = MonthlySoldDf['MonthlySoldFetchedDate'].dt.year
                MonthlySoldDf['Month'] = MonthlySoldDf['MonthlySoldFetchedDate'].dt.month
                
                # Group by Year and Month, then calculate the average
                monthly_average = MonthlySoldDf.groupby(['Year', 'Month'])['MonthlySoldCount'].mean().reset_index()
                monthly_average['MonthlySoldCount'] = monthly_average['MonthlySoldCount'].astype('int', errors='ignore')
                monthly_average = monthly_average.where(monthly_average.notnull(), None)
                
                # Calculate earliest and latest appearances
                MonthlySoldEarliestAppearance = MonthlySoldDf['MonthlySoldFetchedDate'].min()
                MonthlySoldLatestAppearance = MonthlySoldDf['MonthlySoldFetchedDate'].max()
            else:
                # Assign default None values if monthlySoldHistory is None or invalid
                MonthlySoldDf = pd.DataFrame(columns=['MonthlySoldFetchedDate', 'MonthlySoldCount', 'Year', 'Month'])
                monthly_average = pd.DataFrame(columns=['Year', 'Month', 'MonthlySoldCount'])
                MonthlySoldEarliestAppearance = None
                MonthlySoldLatestAppearance = None

            # Update the product DataFrame
            product_df['analysis.MonthlySoldHistAnalysis'] = [monthly_average.to_json(orient='records')] * len(product_df)
            product_df['analysis.MonthlySoldEarliestAppearance'] = [MonthlySoldEarliestAppearance]
            product_df['analysis.MonthlySoldLatestAppearance'] = [MonthlySoldLatestAppearance]

            # Convert date columns to string type
            product_df['analysis.MonthlySoldEarliestAppearance'] = product_df['analysis.MonthlySoldEarliestAppearance'].astype('str')
            product_df['analysis.MonthlySoldLatestAppearance'] = product_df['analysis.MonthlySoldLatestAppearance'].astype('str')

            ### Sales Estimation by ASIN from ProfitGuru

            url = f"https://www.profitguru.com/ext/api/asin/{asin}?re=0"

            headers = {
                "accept": "application/json, text/plain, */*",
                "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
                "priority": "u=1, i",
                "referer": "https://www.profitguru.com/calculator/sales",
                "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"macOS"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "x-app-type": "calc"
            }

            response = requests.get(url, headers=headers)

            sales_est = json.loads(response.text)

            url = f"https://www.profitguru.com/api/product/{sales_est.get('product').get('id')}/history/data"


            response = requests.get(url, headers=headers)

            sales_est_B = json.loads(response.text)
            sales_est_B.get('data').get('fees')

            product_df['analysis.ProfitGuruSalesEstimate']       = [sales_est.get('product').get('sales30')]
            product_df['caculation.ProfitGuruTotalFBACost']      =  [sales_est_B.get('data').get('fees').get('total')]
            product_df['caculation.ProfitGuruFBAFulfilmentCost'] =  [sales_est_B.get('data').get('fees').get('fba')]
            product_df['caculation.ProfitGuruAmazonReferalCost'] =  [sales_est_B.get('data').get('fees').get('ref')]
            product_df['caculation.ProfitGuruAmazonStorageCost'] =  [sales_est_B.get('data').get('fees').get('storage')]



            credentials=dict(
                    refresh_token=SP_API_REFRESH_TOKEN,
                    lwa_app_id=LWA_APP_ID,
                    lwa_client_secret=LWA_CLIENT_SECRET
                )
            products = Products(refresh_token=SP_API_REFRESH_TOKEN,
                                credentials=credentials).get_item_offers(asin,item_condition='New')
            product = json.loads(json.dumps(products.payload))
            product = pd.DataFrame(product.get('Offers'))

            product = product[['SellerId','ShippingTime','IsBuyBoxWinner','IsFulfilledByAmazon']]

            seller_id = fba_sellers.loc[fba_sellers['sellerId'].apply(lambda x: isinstance(x, str)), 'sellerId'].tolist()


            FBASellerShippingDetails = []

            # Loop through each seller_id
            for seller_id in seller_id:
                # Filter the product DataFrame for the current seller_id
                filtered_product = product[product['SellerId'] == seller_id]
                
                # Extract 'ShippingTime' values and get the availabilityType and availableDate
                if not filtered_product.empty:  # Ensure the filtered dataframe is not empty
                    shipping_time_dict = filtered_product['ShippingTime'].iloc[0]  # Access the first match (modify if there are multiple)
                    
                    availability_type = shipping_time_dict.get('availabilityType', None)
                    available_date = shipping_time_dict.get('availableDate', None)
                    minimumHours = shipping_time_dict.get('minimumHours', None)
                    maximumHours = shipping_time_dict.get('maximumHours', None)
                    
                    # Append results
                    FBASellerShippingDetails.append({
                        'SellerId': seller_id,
                        'AvailabilityType': availability_type,
                        'AvailableDate': available_date,
                        'MinimumHours': minimumHours,
                        'MaximumHours': maximumHours
                    })

            # Convert the results list into a DataFrame
            FBASellerShippingDetailsDF = pd.DataFrame(FBASellerShippingDetails)

            # Output the final DataFrame
            FBASellerShippingDetailsDF = FBASellerShippingDetailsDF.where(FBASellerShippingDetailsDF.notnull(), None)

            product_df['analysis.FBASellersShipmentTime'] = [FBASellerShippingDetailsDF.to_json(orient='records')] * len(product_df)


            filter={'Vendor SKU': vendor_sku}
            project={'Promotion Price': 1, '_id': 0}

            result = product_inventory_collection.find(
            filter=filter,
            projection=project)

            for docnt in result:
               cost_price = docnt.get("Promotion Price", "Promotion Price not found")
               cost_price = float(cost_price)

            # Input Data
            cost_price = cost_price
            shipping_to_amazon = 0.5
            referral_fee = sales_est_B.get('data').get('fees').get('ref')
            storage_fee = sales_est_B.get('data').get('fees').get('storage')
            fba_fees = sales_est_B.get('data').get('fees').get('fba')
            sell_price = products.payload.get('Summary').get('LowestPrices')[0].get('LandedPrice').get('Amount')

            # Calculate Metrics
            metrics = calculate_metrics(sell_price, referral_fee, fba_fees, storage_fee, shipping_to_amazon, cost_price)

            product_df['caculation.CostPrice']               =  [cost_price]
            product_df['caculation.OtherCost']               =  [shipping_to_amazon]
            product_df['caculation.AmazonReferralFee']       =  [referral_fee]
            product_df['caculation.AmazonStorageFee']        =  [storage_fee]
            product_df['caculation.FBAfees']                 =  [fba_fees]
            product_df['caculation.MinSellingPrice']         =  [sell_price]
            product_df['caculation.PurchaseSellerProceeds']  =  [metrics['Seller Proceeds']]
            product_df['caculation.PurchaseNetProfit']       =  [metrics['Net Profit']]
            product_df['caculation.PurchaseNetProfitMargin'] =  [metrics['Net Margin %']]
            product_df['caculation.PurchaseROI']             =  [metrics['ROI %']]


            client = connect_to_mongodb()
            product_analysis_collection.insert_one(json.loads(product_df.to_json(orient='records'))[0])

            print('Successufully Added the records')
    except Exception as e:
        print(f"Error in the {asin} - {e}")
