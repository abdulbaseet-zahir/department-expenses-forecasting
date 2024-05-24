from fastapi import FastAPI
import uvicorn

import pandas as pd
from prophet import Prophet

data_path = "data/model.csv"


def get_data(data_path):
    data_df = pd.read_csv(data_path)
    data_df["Months"] = pd.to_datetime(data_df["Months"])
    data_df = (
        data_df.groupby(["Department_ID", "Months"])["Expenses"].sum().reset_index()
    )
    data_df = data_df.pivot(
        index="Months", columns="Department_ID", values="Expenses"
    ).reset_index()

    return data_df


def get_one_year_forcast(ID):
    data_df = get_data(data_path)
    data = data_df[["Months", ID]].rename(columns={"Months": "ds", ID: "y"})

    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]


def get_all_expenses(ID):
    data_df = get_data(data_path)
    data = data_df[["Months", ID]].rename(columns={"Months": "ds", ID: "y"})

    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    future["y"] = data["y"]
    future["yhat"] = forecast["yhat"]
    future.fillna(0, inplace=True)
    return future


app = FastAPI()


@app.get("/get_one_year_forcast/{ID}")
async def square(ID: int):
    return get_one_year_forcast(ID).to_dict(orient="records")


@app.get("/get_all_expenses/{ID}")
async def square(ID: int):
    return get_all_expenses(ID).to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
    )
