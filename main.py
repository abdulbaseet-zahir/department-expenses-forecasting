from fastapi import FastAPI
import uvicorn
import pandas as pd
from prophet import Prophet

data_path = "data/data.csv"

app = FastAPI()


def get_data(data_path: str) -> pd.DataFrame:
    data_df = pd.read_csv(data_path)
    data_df["Months"] = pd.to_datetime(data_df["Months"])
    data_df["Unit_ID"] = data_df["Unit_ID"].map(lambda x: int(str(x)[2:]))
    return data_df


def prepare_data(
    data_df: pd.DataFrame, department_id: int, unit_id: int = None
) -> pd.DataFrame:
    if unit_id:
        grouped_df = (
            data_df.groupby(["Department_ID", "Unit_ID", "Months"])["Expenses"]
            .sum()
            .reset_index()
        )
        pivoted_df = grouped_df.pivot(
            index="Months", columns=["Department_ID", "Unit_ID"], values="Expenses"
        ).reset_index()
        months = pivoted_df["Months"]
        department_data = pd.concat([months, pivoted_df[department_id]], axis=1)
        return department_data[["Months", unit_id]].rename(
            columns={"Months": "ds", unit_id: "y"}
        )
    else:
        grouped_df = (
            data_df.groupby(["Department_ID", "Months"])["Expenses"].sum().reset_index()
        )
        pivot_df = grouped_df.pivot(
            index="Months", columns="Department_ID", values="Expenses"
        ).reset_index()
        return pivot_df[["Months", department_id]].rename(
            columns={"Months": "ds", department_id: "y"}
        )


def create_and_predict_model(data: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]


# optional unit id
@app.get("/get_department_expense_forecasts/{department_id}")
async def get_department_expense_forecasts(
    department_id: int, unit_id: int = None
) -> list[dict]:
    data_df = get_data(data_path)
    data = prepare_data(data_df, department_id, unit_id)
    forecast = create_and_predict_model(data)

    future_data = forecast.merge(data, on="ds", how="left")
    # future_data.fillna(0, inplace=True)
    future_data.rename(
        columns={"ds": "Months", "y": "Actuals", "yhat": "Forecast"}, inplace=True
    )
    return future_data.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
