import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

flo_data = pd.read_csv("flo_data.csv")
df = flo_data.copy()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for i in [col for col in df.columns if "total" in col]:
    replace_with_thresholds(df, i)

df["total_num_orders"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_deposit"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df[["total_num_orders", "total_deposit"]].head()

df[[col for col in df.columns if "date" in col]].info()
date_columns_names = [col for col in df.columns if "date" in col]
for j in date_columns_names:
    df[j] = pd.to_datetime(df[j])

df.info()
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

cltv_df = df.groupby('master_id').agg(
    {'last_order_date': lambda x: x,
     "first_order_date": lambda i: i,
     'total_num_orders': lambda j: j,
     'total_deposit': lambda k: k})

cltv_df.info()

cltv_df = cltv_df[cltv_df["total_num_orders"] > 1]

cltv_df["recency_1"] = (today_date - cltv_df["first_order_date"]).astype("timedelta64[D]")
cltv_df["recency_2"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).astype("timedelta64[D]")

cltv_df["recency_1"] = cltv_df["recency_1"] / 7
cltv_df["recency_2"] = cltv_df["recency_2"] / 7

cltv_df["pur_freq"] = cltv_df["total_num_orders"] # / len(cltv_df)
cltv_df["avg_monetary"] = cltv_df["total_deposit"] / cltv_df["total_num_orders"]

cltv_df.reset_index(inplace=True)

cltv_df = cltv_df[["master_id", "recency_1", "recency_2", "pur_freq", "avg_monetary"]]

cltv_df.columns = ["customer_id", "T_weekly", "recency_cltv_weekly", "frequency", "monetary_cltv_avg"]

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month", ascending=False)

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df["final"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.rename(columns={"final" : "Segment"}, inplace=True)

cltv_df.sort_values("Segment", ascending=False)

