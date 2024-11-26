import streamlit as st
from snowflake.snowpark.context import get_active_session
import snowflake.snowpark.functions as F
import json
from snowflake.snowpark.functions import *
from snowflake.ml.modeling.preprocessing import label_encoder, MinMaxScaler
from snowflake.snowpark.functions import rank, col
from snowflake.snowpark.window import Window
import pandas as pd
import time

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
session = get_active_session()

sparse_features = ['MENU_ITEM_NAME', 
                   'MENU_TYPE', 
                   'TRUCK_BRAND_NAME', 
                   'ITEM_CATEGORY', 
                   'ITEM_SUBCATEGORY',
                   'CITY',
                   'COUNTRY',
                   'GENDER',
                   'MARITAL_STATUS',]

dense_features = ['SALE_PRICE_USD',
                  'AGE',
                  'AVG_YEARLY_PURCHASE_AMOUNT',
                  'AVG_MONTHLY_PURCHASE_AMOUNT',
                  'AVG_WEEKLY_PURCHASE_AMOUNT']

sparse_features_encoded = ['MENU_ITEM_NAME_ENCODED',
                           'MENU_TYPE_ENCODED',
                           'TRUCK_BRAND_NAME_ENCODED',
                           'ITEM_CATEGORY_ENCODED',
                           'ITEM_SUBCATEGORY_ENCODED',
                           'CITY_ENCODED',
                           'COUNTRY_ENCODED',
                           'GENDER_ENCODED',
                           'MARITAL_STATUS_ENCODED']

def get_serialized_label_encoders():
    stage_name = 'ml.UDF_STAGE'
    dir_name='dlrm_label_encoders'
    file_name = 'label_encoders.json'
    
    session.file.get(f'@{stage_name}/{dir_name}/{file_name}',"/tmp/dir/")
    modified_label_encoders = {}
    # Read the downloaded file content
    with open(f"/tmp/dir/label_encoders.json", "r") as f:
        serialized_content = f.read()
    
    # Deserialize the label encoders
    serialized_label_encoders = json.loads(serialized_content)
    for feat, params in serialized_label_encoders.items():
        modified_params = params.copy()  # Create a copy to avoid modifying the original dictionary
        # Update the classes_ list to create the desired output structure
        modified_params['classes_'] = {i: v for i, v in enumerate(params['classes_'])}
        modified_label_encoders[params['output_cols'][0]] = modified_params['classes_']
    return modified_label_encoders
    
def apply_label_encoding(df):
    if 'modified_label_encoders' not in st.session_state:
        st.session_state.modified_label_encoders = get_serialized_label_encoders()
    serialized_label_encoders = st.session_state.modified_label_encoders
    input_cols_to_drop = []
    for feat, mapping in serialized_label_encoders.items():
        # Get the input column from the serialized label encoder
        input_col = feat[:-len("_ENCODED")]  # Extract the column name from the encoded column name

        # Get the mapping dictionary for the current feature
        encoding_mapping = {int(k): v for k, v in mapping.items()}
        
        # Generate encoding for the input column
        case_stmt = "CASE"
        for encoded_val, class_val in encoding_mapping.items():
            case_stmt += f" WHEN {input_col} = '{class_val}' THEN {encoded_val}"
        case_stmt += " ELSE NULL END"

        # Apply label encoding using the CASE statement
        output_col = f"{input_col}_encoded"
        df = df.withColumn(output_col, F.expr(case_stmt))

        if input_col not in ("CITY", "TRUCK_BRAND_NAME", "MENU_ITEM_NAME"):
            input_cols_to_drop.append(input_col)

    # Drop the original input columns
    df = df.drop(*input_cols_to_drop)
    return df

def get_additional_columns_needed():
    if 'avg_yearly_purchase_amount' not in st.session_state:
        st.session_state.avg_yearly_purchase_amount = session.table("ANALYTICS.ORDERS_V") \
                                                                .groupBy("customer_id") \
                                                                .agg(sum("order_total").alias("total_order"),
                                                                (expr("TIMESTAMPDIFF(YEAR, MIN(date), MAX(date))") + 1).alias("year_diff")) \
                                                                .selectExpr("customer_id", "ROUND(total_order / year_diff, 2) AS avg_yearly_purchase_amount")

    if 'avg_weekly_purchase_amount' not in st.session_state:
        st.session_state.avg_weekly_purchase_amount = session.table("ANALYTICS.ORDERS_V") \
                                                                .groupBy("customer_id") \
                                                                .agg(sum("order_total").alias("total_order"),
                                                                (expr("TIMESTAMPDIFF(WEEK, MIN(date), MAX(date))") + 1).alias("week_diff")) \
                                                                .selectExpr("customer_id", "ROUND(total_order / week_diff, 2) AS avg_weekly_purchase_amount") 
    
    if 'avg_monthly_purchase_amount' not in st.session_state:
        st.session_state.avg_monthly_purchase_amount = session.table("ANALYTICS.ORDERS_V") \
                                                                .groupBy("customer_id") \
                                                                .agg(sum("order_total").alias("total_order"),
                                                                     (expr("TIMESTAMPDIFF(MONTH, MIN(date), MAX(date))") + 1).alias("month_diff")) \
                                                                .selectExpr("customer_id", "ROUND(total_order / month_diff, 2) AS avg_monthly_purchase_amount")
    
    cust_avgs_spdf = st.session_state.avg_monthly_purchase_amount \
                    .join(st.session_state.avg_weekly_purchase_amount, on="CUSTOMER_ID") \
                    .join(st.session_state.avg_yearly_purchase_amount, on="CUSTOMER_ID")
    return cust_avgs_spdf
    

# Function to get filtered Data
def get_filtered_data():
    with st.spinner("Getting Filtered Data..."):
        time.sleep(1)
        cust_avgs = get_additional_columns_needed()
        item_df = session.table('ml.menu_item_features').filter("item_category != 'Beverage'")
        customer_df = session.table('ml.customer_features')
        purchase_df = session.table('analytics.loyalty_purchased_items')
        if st.session_state.filters['trucks_not_visited']:
            all_customer_ids_and_trucks = purchase_df \
            .join(item_df, purchase_df.menu_item_name == item_df.menu_item_name) \
                                    .select(purchase_df.customer_id, item_df.truck_brand_name, purchase_df.purchased) \
                                    .distinct()
            
            customer_visited_trucks = purchase_df \
                                .join(item_df, purchase_df.menu_item_name == item_df.menu_item_name) \
                                .filter(purchase_df.purchased == 1) \
                                .select(purchase_df.customer_id, item_df.truck_brand_name, purchase_df.purchased) \
                                .distinct()

            never_visited_trucks = all_customer_ids_and_trucks \
                                        .join(customer_visited_trucks, 
                                              (all_customer_ids_and_trucks.customer_id == customer_visited_trucks.customer_id) & 
                                              (all_customer_ids_and_trucks.truck_brand_name == customer_visited_trucks.truck_brand_name),
                                              "left_anti")

            item_df_df = item_df.toDF(["menu_type", "truck_brand_name", "menu_item_name", "item_category", "item_subcategory", "sale_price_usd"])
            item_df_df_renamed = item_df_df.withColumnRenamed("truck_brand_name", "item_truck_brand_name")
            
            # Joining DataFrames
            result_df = never_visited_trucks \
                        .join(item_df_df_renamed, never_visited_trucks.truck_brand_name == item_df_df_renamed.item_truck_brand_name) \
                        .select(never_visited_trucks["*"], item_df_df_renamed["*"])
            df = result_df
        else:
            df = purchase_df.join(item_df, 'menu_item_name', 'left')
        filters = get_filters()
        result_df = df.join(customer_df, 'customer_id', 'left')
        final_df = result_df.join(cust_avgs, 'customer_id', 'left').filter(filters).order_by(asc("customer_id")).limit(10000)
    return final_df

# Function to simulate processing features
def process_features(df):
    with st.spinner("Processing Features..."):
        time.sleep(1)
        df_encoded = apply_label_encoding(df)
        mms = MinMaxScaler(feature_range=(0, 1), input_cols=dense_features, output_cols=dense_features)
        mms.fit(df_encoded)
        testdata = mms.transform(df_encoded)
    return testdata
        

# Function to simulate inference on the deep learning model
def infer_model(test_data):
    with st.spinner("Inferring Deep Learning Model on SPCS..."):
        time.sleep(1)
        recommendations_df = test_data.select(
                        "customer_id",
                        "city",
                        "truck_brand_name",
                        F.col("menu_item_name").alias("menu_item_recommendations"),
                        "purchased",
                        F.call_udf(
                            "REGISTRY.TB_REC_SERVICE_DEMO_PREDICT_FORWARD", 
                            F.array_construct(*sparse_features_encoded),
                            F.array_construct(*dense_features),
                        )["output_feature_0"][0].alias("prediction")
                    ).order_by("prediction", ascending=0)
        
        # Define a window partitioned by customer_id and ordered by prediction score
        windowSpec = Window.partitionBy(col("customer_id")).orderBy(col("prediction").desc())
            
        # Add a rank column to rank the predictions for each customer
        rankedDf = recommendations_df.withColumn("rank", rank().over(windowSpec))
        
        # Filter for rows where rank is 1 (top prediction for each customer)
        # topPredictionsDf = rankedDf.filter(col("rank") == 1)

        # Filter for rows where rank is less than or equal to 3 (top 3 predictions for each customer)
        topPredictionsDf = rankedDf.filter(col("rank") <= 3)
        
        # Convert DataFrame to Pandas DataFrame
        top_predictions_pd = topPredictionsDf.toPandas()
        
        # Group by customer_id and collect the top 3 recommendations as a list
        grouped_top_predictions = top_predictions_pd.groupby("CUSTOMER_ID").agg({'CITY': 'first','MENU_ITEM_RECOMMENDATIONS': lambda x: list(x)}).reset_index()
        
        distinct_customer_ids = grouped_top_predictions['CUSTOMER_ID'].unique()
        sql_in_clause = ""
        
        # Loop through the distinct customer_ids to form the SQL IN clause
        for customer_id in distinct_customer_ids:
            # Append each customer_id to the SQL IN clause
            sql_in_clause += "'" + str(customer_id) + "', "
        
        # Remove the trailing comma and space
        sql_in_clause = sql_in_clause[:-2]
        history_df = session.sql(f"""select customer_id, 
                                    menu_item_name as Purchase_History
                                    from analytics.loyalty_purchased_items
                                    where purchased = 1
                                    and customer_id in ({sql_in_clause});""").to_pandas()
        grouped_history_df = history_df.groupby('CUSTOMER_ID')['PURCHASE_HISTORY'].agg(list).reset_index()

        # Merge finalDf_pd with grouped_history_df on customer_id
        finalDf_pd = pd.merge(grouped_top_predictions, grouped_history_df, on='CUSTOMER_ID', how='left')

        # Reorder columns
        finalDf_pd = finalDf_pd[['CUSTOMER_ID', 'CITY', 'PURCHASE_HISTORY', 'MENU_ITEM_RECOMMENDATIONS']]

        finalDf_pd['PURCHASE_HISTORY'] = finalDf_pd['PURCHASE_HISTORY'].apply(lambda x: x if isinstance(x, list) else [])
        finalDf_pd = finalDf_pd.sort_values(by='CUSTOMER_ID', ascending=True)
    return finalDf_pd

# Function to simulate saving recommendations
def save_recommendations():
    st.session_state.save_clicked = True
    st.session_state.details_changed = True

def get_filters():
    filter_clause = ' 1=1 '
    if st.session_state.filters['country'] is not None and len(st.session_state.filters['country']) > 0:
        country_where_clause = "("
        for x in st.session_state.filters['country']:
            country_where_clause += f"'{x}',"
        country_where_clause = country_where_clause[:-1] + ")"
        filter_clause += " and country in " + country_where_clause

    if st.session_state.filters['city'] is not None and len(st.session_state.filters['city']) > 0:
        city_where_clause = "("
        for x in st.session_state.filters['city']:
            city_where_clause += f"'{x}',"
        city_where_clause = city_where_clause[:-1] + ")"
        filter_clause += " and city in " + city_where_clause

    if st.session_state.filters['truck_brand_name'] is not None and len(st.session_state.filters['truck_brand_name']) > 0:
        truck_brand_name_where_clause = "("
        for x in st.session_state.filters['truck_brand_name']:
            truck_brand_name_where_clause += f"'{x}',"
        truck_brand_name_where_clause = truck_brand_name_where_clause[:-1] + ")"
        filter_clause += " and truck_brand_name in " + truck_brand_name_where_clause

    if st.session_state.filters['menu_items_not_ordered'] == True:
        filter_clause += " and purchased = 0"
    return filter_clause

def on_change_callback(key):
    st.session_state.filters[key] = st.session_state[key]
    if key == "country":
        st.session_state.filters['city'] = []
        country_where_clause = ""
        if len(st.session_state.filters['country']) > 0 :
            country_where_clause += "("
            for x in st.session_state.filters['country']:
                country_where_clause += f"'{x}',"
            country_where_clause = country_where_clause[:-1] + ")"
            st.session_state.city_list = session.sql(f"""select distinct CITY 
                                                        from ml.customer_features 
                                                        where COUNTRY in {country_where_clause} 
                                                        order by CITY;""").to_pandas()
        else:
            st.session_state.city_list = session.sql("""select distinct CITY 
                                                        from ml.customer_features 
                                                        order by CITY;""").to_pandas()

def get_recommendations():
    st.session_state.get_recommendations_clicked = True
    st.session_state.filters_changed = True
    st.session_state.details_changed = True

if 'save_clicked' not in st.session_state:
    st.session_state.save_clicked = False

if 'get_recommendations_clicked' not in st.session_state:
    st.session_state.get_recommendations_clicked = False

if 'filters_changed' not in st.session_state:
    st.session_state.filters_changed = False 

if 'details_changed' not in st.session_state:
    st.session_state.details_changed = False 

st.title(":truck: Tasty Bytes Menu Item Recommendations :truck:")
st.markdown("**:black[Customer & Menu Filters]**")

st.session_state.filters = {
                            'country': None,
                            'city': None,
                            'truck_brand_name': None,
                            'trucks_not_visited': False,
                            'menu_items_not_ordered': False
                           }

col1, col2, col3 = st.columns(3, gap="medium")
if 'country_list' not in st.session_state:
    st.session_state.country_list = session.sql("""select distinct COUNTRY
                                                from ml.customer_features 
                                                order by COUNTRY;""").to_pandas()
country_list = st.session_state.country_list['COUNTRY'].tolist()
st.session_state.filters['country'] = col1.multiselect('Country',
                                                       country_list,
                                                       default=st.session_state.filters['country'],
                                                       on_change=on_change_callback,
                                                       key="country",
                                                       kwargs={"key": "country"})

if 'city_list' not in st.session_state:
    st.session_state.city_list = session.sql("""select distinct CITY 
                                                from ml.customer_features 
                                                order by CITY;""").to_pandas()
city_list = st.session_state.city_list['CITY'].tolist()
st.session_state.filters['city'] = col2.multiselect('City',
                                                    city_list,
                                                    default=st.session_state.filters['city'],
                                                    on_change=on_change_callback,
                                                    key="city",
                                                    kwargs={"key": "city"})

if 'truck_brand_name_list' not in st.session_state:
    st.session_state.truck_brand_name_list = session.sql("""select distinct TRUCK_BRAND_NAME
                                                        from ml.menu_item_features
                                                        order by TRUCK_BRAND_NAME;""").to_pandas()
truck_brand_name_list = st.session_state.truck_brand_name_list['TRUCK_BRAND_NAME'].tolist()
st.session_state.filters['truck_brand_name'] = col3.multiselect('Truck Brand Name',
                                                      truck_brand_name_list,
                                                      default=st.session_state.filters['truck_brand_name'],
                                                      on_change=on_change_callback,
                                                      key="truck_brand_name",
                                                               kwargs={"key": "truck_brand_name"})

st.markdown("**:black[Additional Filter Options]**")  
st.session_state.filters['trucks_not_visited'] = st.checkbox('Trucks not visited',
                                                               value=st.session_state.filters['trucks_not_visited'],
                                                               key="trucks_not_visited",
                                                               on_change=on_change_callback,
                                                            kwargs={"key": "trucks_not_visited"})

st.session_state.filters['menu_items_not_ordered'] = st.checkbox('Menu items not ordered',
                                                                 value=st.session_state.filters['menu_items_not_ordered'],
                                                                 key="menu_items_not_ordered",
                                                                 on_change=on_change_callback,
                                                                kwargs={"key": "menu_items_not_ordered"})

    
st.button("**:blue[Get Recommendations]**", on_click=get_recommendations)
if st.session_state.get_recommendations_clicked:
    if st.session_state.filters_changed:
        df = get_filtered_data()
        st.session_state.df_count = df.count()
        if st.session_state.df_count == 0:
            st.warning("Customers May have visited all the trucks or tried all the menu items, Please select different filters.")
        else:
            data = process_features(df)
            st.session_state.recommendations = infer_model(data)
            st.session_state.filters_changed = False
    if st.session_state.details_changed:
        col1, col2 = st.columns([12,1])
        if st.session_state.df_count > 0:
            col1.markdown("**Top Recommended Menu Items**")
            col2.button("**:blue[Save]**", on_click=save_recommendations)
            if st.session_state.save_clicked:
                with st.spinner("Saving Recommendations..."):
                    # st.session_state.recommendations.write.mode("overwrite").save_as_table("recommendations")
                    session.write_pandas(st.session_state.recommendations, "recommendations", auto_create_table=True, overwrite=True)
                    st.success("Recommendations Saved Successfully!")
                    st.session_state.save_clicked = False
            st.dataframe(st.session_state.recommendations, use_container_width=True)
            st.session_state.details_changed = False

