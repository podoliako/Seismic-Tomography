import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dwh import get_tp_ts, pd, load_to_postgres, load_to_postgis
from utilities import make_grid_3d_time, generate_name
import json


def estimate_ts_tp_ratio(lon, lat, r_events, r_stations, dttm_from, dttm_to):
    df = get_tp_ts(lon, lat, r_events, r_stations, dttm_from, dttm_to)
    if df.empty:
        return {
        "result":"empty_df" 
        } , None


    df["delta_t_p"] = df["delta_t_p"].dt.total_seconds()
    df["delta_t_s"] = df["delta_t_s"].dt.total_seconds()

    df = df[(df["delta_t_p"] > 0) & (df["delta_t_s"] > 0)].reset_index(drop=True)

    df["ratio_ts_tp"] = df["delta_t_s"] / df["delta_t_p"]

    X = df[["delta_t_p"]].values
    y = df["delta_t_s"].values

    model_free = LinearRegression()
    try:
        model_free.fit(X, y)
    except Exception as e:
        return {"error": {e}}
    k_free = float(model_free.coef_[0])
    b_free = float(model_free.intercept_)
    rmse_free = float(np.sqrt(mean_squared_error(y, model_free.predict(X))))

    model_zero = LinearRegression(fit_intercept=False)
    model_zero.fit(X, y)
    k_zero = float(model_zero.coef_[0])
    rmse_zero = float(np.sqrt(mean_squared_error(y, model_zero.predict(X))))

    return {
        "regression_free": {
            "k": k_free,
            "b": b_free,
            "rmse": rmse_free
        },
        "regression_zero_intercept": {
            "k": k_zero,
            "rmse": rmse_zero
        }
    }, df

def run_experiment_vp_vs(
    lat_min, 
    lat_max, 
    lon_min, 
    lon_max, 
    n_steps_lat, 
    n_steps_lon,
    r_events,
    r_stations,
    dttm_from, 
    dttm_to, 
    n_steps_time
    ): # Добавить красивую загрузку 

    experiment_nm = generate_name(['exp', dttm_from, dttm_to, lat_min, lon_min])
    
    grid = make_grid_3d_time(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        n_steps_lat=n_steps_lat,
        n_steps_lon=n_steps_lon,
        dttm_from=dttm_from,
        dttm_to=dttm_to,
        n_steps_time=n_steps_time
        )

    rows = []
    time_travel_df = None
    for point in grid:
        params, tt_df = estimate_ts_tp_ratio(
            lat=point["lat"],
            lon=point["lon"],
            r_events=r_events,
            r_stations=r_stations,
            dttm_from=point["t_from"],
            dttm_to=point["t_to"]              
            )   
        
        if time_travel_df is None and tt_df is not None and tt_df.shape[0] >= 1:
            time_travel_df = tt_df
        elif time_travel_df is not None and tt_df is not None:
            time_travel_df = pd.concat([time_travel_df, tt_df], ignore_index=True)
        else:
            print('empty_df')

        rows.append({
            "experiment_nm": experiment_nm,
            "lat": float(point["lat"]),
            "lon": float(point["lon"]),
            "r_events": float(r_events),
            "r_stations": float(r_stations),
            "t_from": str(point["t_from"]),
            "t_to": str(point["t_to"]),
            "params": json.loads(json.dumps(params, default=list))
        })
        
    exp_df = pd.DataFrame(rows)
    time_travel_df['experiment_nm'] = experiment_nm
    time_travel_df = time_travel_df.drop(['event_dttm', 'ratio_ts_tp'], axis=1)

    load_to_postgis(exp_df, 'experiments')
    load_to_postgres(time_travel_df, 'travel_times')

    return exp_df, time_travel_df
