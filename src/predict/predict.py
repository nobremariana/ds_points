#%%
import pandas as pd
import sqlalchemy

model_series = pd.read_pickle("../../models/model_churn.pkl")
model_series
# %%
engine = sqlalchemy.create_engine("sqlite:///../../data/feature_store.db")

with open("etl.sql", "r") as open_file:
    query = open_file.read()

df = pd.read_sql(query, engine)

pred = model_series['model'].predict_proba(df[model_series['features']])
proba_churn = pred[:,1]

df_predict = df[['dtRef', 'idCustomer']].copy()
df_predict['prob_churn'] = proba_churn.copy()

df_predict = (df_predict.sort_values("prob_churn", ascending=False)
                        .reset_index(drop=True))
df_predict
# %%
