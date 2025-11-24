import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
import requests
import pandas as pd
from geopandas import *
from shapely.geometry import Point
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Имя того, кто запускает код, чтобы можно было различать логи
RUNNER_NAME = 'Egor'

def add_runner_name(params_json):
    if isinstance(params_json, str):
        params = json.loads(params_json)
    else:
        params = params_json

    params["runner_name"] = RUNNER_NAME
    return params

def make_json_serializable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except:
        return None

def format_region(field_nm, value):
    return f"{field_nm}={value}&" if value else ""

def get_quake_ml(url):
    return requests.get(url).text

def extract_events(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}

    events = []

    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')
        event_type = event.find('bed:type', ns).text if event.find('bed:type', ns) is not None else None
        event_type_certainty = event.find('bed:typeCertainty', ns).text if event.find('bed:typeCertainty', ns) is not None else None
        origin = event.find('bed:origin', ns)

        if origin is not None:
            time = origin.find('.//bed:time/bed:value', ns).text if origin.find('.//bed:time/bed:value', ns) is not None else None
            lat = float(origin.find('.//bed:latitude/bed:value', ns).text) if origin.find('.//bed:latitude/bed:value', ns) is not None else None
            lon = float(origin.find('.//bed:longitude/bed:value', ns).text) if origin.find('.//bed:longitude/bed:value', ns) is not None else None
            depth = float(origin.find('.//bed:depth/bed:value', ns).text) if origin.find('.//bed:depth/bed:value', ns) is not None else None
        else:
            time = lat = lon = depth = None

        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "event_type_certainty": event_type_certainty,
            "lon": lon,
            "lat": lat,
            "depth": depth,
            "event_dttm": time
        })

    df = pd.DataFrame(events)

    geom = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df, crs=crs, geometry=geom)
    df = gdf.rename({'geometry': 'loc'}, axis='columns')
    df = df.set_geometry("loc")

    return df

def extract_arrivals(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}
    
    arrivals_dict = {}
    
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')
        
        for arrival in event.findall('.//bed:arrival', ns):
            phase = arrival.find('bed:phase', ns).text if arrival.find('bed:phase', ns) is not None else None
            pick_id = arrival.find('bed:pickID', ns).text if arrival.find('bed:pickID', ns) is not None else None
            
            if pick_id:
                pick = root.find(f".//bed:pick[@publicID='{pick_id}']", ns)
                pick_id = pick_id.replace('smi:ISC/pickid=', '')
                if pick is not None:
                    arrival_time = pick.find('bed:time/bed:value', ns).text if pick.find('bed:time/bed:value', ns) is not None else None
                    waveform = pick.find('bed:waveformID', ns)
                    
                    if waveform is not None:
                        network_nm = waveform.get('networkCode')
                        station_nm = waveform.get('stationCode')
                        
                        key = (event_id, station_nm, network_nm, pick_id, phase)
                        
                        if key not in arrivals_dict or (arrival_time and arrival_time < arrivals_dict[key]):
                            arrivals_dict[key] = arrival_time
    
    arrivals = [
        {
            "event_id": k[0],
            "station_nm": k[1],
            "network_nm": k[2],
            "pick_id": k[3],
            "arrival_type": k[4],
            "arrival_dttm": v
        }
        for k, v in arrivals_dict.items()
    ]
    
    df = pd.DataFrame(arrivals)
    return df

def extract_magnitudes(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}
    
    magnitudes = []
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')
        
        # Find all magnitude elements for this event
        for mag in event.findall('bed:magnitude', ns):
            mag_value = mag.find('bed:mag/bed:value', ns).text if mag.find('bed:mag/bed:value', ns) is not None else None
            mag_type = mag.find('bed:type', ns).text if mag.find('bed:type', ns) is not None else None
            origin_id = mag.get('publicID', '').replace('smi:ISC/magid=', '')
            
            # Get creation info if available
            creation_info = mag.find('bed:creationInfo', ns)
            author = creation_info.find('bed:author', ns).text if creation_info is not None and creation_info.find('bed:author', ns) is not None else None

            magnitudes.append({
                "event_id": event_id,
                "magnitude_id": origin_id,
                "mag_value": float(mag_value) if mag_value is not None else None,
                "mag_type": mag_type,
                "author_nm": author,
            })
    
    df = pd.DataFrame(magnitudes)
    return df

def extract_stations(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ns = {'fdsn': 'http://www.fdsn.org/xml/station/1'}
    
    stations = []
    
    for network in root.findall('.//fdsn:Network', ns):
        network_code = network.get('code')
        
        for station in network.findall('.//fdsn:Station', ns):
            station_code = station.get('code')
            lat = station.find('fdsn:Latitude', ns).text if station.find('fdsn:Latitude', ns) is not None else None
            lon = station.find('fdsn:Longitude', ns).text if station.find('fdsn:Longitude', ns) is not None else None
            
            stations.append({
                "station_nm": station_code,
                "network_nm": network_code,
                "lat": float(lat) if lat else None,
                "lon": float(lon) if lon else None
            })
    
    df = pd.DataFrame(stations)
    return df


# ==================== Получает события и магнитуды по прямоугольной области ==================== 
def get_events_and_mags_by_rect(dttm_from, dttm_to, min_mag, lat_min, lat_max, lon_min, lon_max):
    start_dt = dttm_from.split('-')
    end_dt = dttm_to.split('-')
    
    url = f'''https://www.isc.ac.uk/cgi-bin/web-db-run?out_format=CATQuakeML&request=COMPREHENSIVE&
        searchshape=RECT&
        start_year={start_dt[0]}&
        start_month={start_dt[1]}&
        start_day={start_dt[2]}&
        start_time=00:00:00&
        end_year={end_dt[0]}&
        end_month={end_dt[1]}&
        end_day={end_dt[2]}&
        end_time=23:59:59&
        bot_lat={lat_min}&
        top_lat={lat_max}&
        left_lon={lon_min}&
        right_lon={lon_max}&
        min_mag={min_mag}'''
    
    url = url.replace('\n', '').replace('        ', '')
    print(f"Requesting: {url}")
    
    xml = get_quake_ml(url)
    
    events = extract_events(xml)
    magnitudes = extract_magnitudes(xml)
    
    return events, magnitudes

# ==================== Получает arrivals по прямоугольной области ==================== 
def get_arrivals_by_rect(dttm_from, dttm_to, min_mag, lat_min, lat_max, lon_min, lon_max):

    start_dt = dttm_from.split('-')
    end_dt = dttm_to.split('-')
    
    url = f'''https://www.isc.ac.uk/cgi-bin/web-db-run?out_format=QuakeML&request=STNARRIVALS&
        searchshape=RECT&
        start_year={start_dt[0]}&
        start_month={start_dt[1]}&
        start_day={start_dt[2]}&
        start_time=00:00:00&
        end_year={end_dt[0]}&
        end_month={end_dt[1]}&
        end_day={end_dt[2]}&
        end_time=23:59:59&
        bot_lat={lat_min}&
        top_lat={lat_max}&
        left_lon={lon_min}&
        right_lon={lon_max}&
        min_mag={min_mag}'''
    
    url = url.replace('\n', '').replace('        ', '')
    print(f"Requesting arrivals: {url}")
    
    return extract_arrivals(get_quake_ml(url))

# ========= Универсальная загрузка данных в базу =========
def load_to_postgres(df, table_name):
    if 'loc' in df.columns:
        df['loc'] = df['loc'].apply(lambda g: g.wkb_hex)

    engine = create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    rows = df.to_dict(orient='records')

    stmt = insert(table).values(rows)
    stmt = stmt.on_conflict_do_nothing()

    with engine.begin() as conn:
        conn.execute(stmt)

    print(f"[{table_name}] Inserted non-conflicting rows successfully.")

# ========= Для отправки логов =========
def post_log(log_nm, start_dt=None, end_dt=None, load_status=None, params=None):
    params = add_runner_name(params)
    log = {
        "log_nm": log_nm,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "params": params,
        "load_status": load_status,
        "log_dttm": datetime.now()
    }
    
    df = pd.DataFrame([log])
    print(df)
    df["params"] = df["params"].apply(make_json_serializable)

    load_to_postgres(df, 'logs')


# ========= Итоговая функция для событий и магнитуд =========
def load_recursion_events_and_mags(dttm_from, dttm_to,
                                   lat_min, lat_max,
                                   lon_min, lon_max,
                                   min_mag=0.5,
                                   depth=0, max_depth=7):
    
    region_id = f"[lat:{lat_min:.2f}-{lat_max:.2f}, lon:{lon_min:.2f}-{lon_max:.2f}]"
    region_params = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "dttm_from": dttm_from,
        "dttm_to": dttm_to,
        "min_mag": min_mag,
        "depth": depth
    }

    if depth >= max_depth:
        print(f"{'  ' * depth} Max recursion depth reached for {region_id}")
        post_log('load_events_mags', dttm_from, dttm_to, 'max_depth', params=region_params)
        return

    post_log('load_events_mags', dttm_from, dttm_to, 'start', params=region_params)
    print(f"{'  ' * depth} Loading region {region_id}")

    events_status = None
    mags_status = None

    # ========== ЗАГРУЗКА EVENTS & MAGNITUDES ==========
    try:
        print(f"{'  ' * depth} Requesting events & mags...")

        events, mags = get_events_and_mags_by_rect(
            dttm_from, dttm_to, min_mag,
            lat_min, lat_max, lon_min, lon_max
        )

        # ——— ПУСТОЙ ОТВЕТ ———
        if len(events) == 0:
            print(f"{'  ' * depth} No events found")
            post_log('load_events_mags', dttm_from, dttm_to,
                     'success_no_data', params=region_params)

            events_status = 'success_no_data'
            mags_status = 'success_no_data'

        else:
            print(f"{'  ' * depth} Loaded {len(events)} events, {len(mags)} magnitudes")
            post_log('load_events_mags', dttm_from, dttm_to,
                     'http_success', params=region_params)

            # ——— ЗАГРУЗКА В БД ———
            try:
                load_to_postgres(events, 'events')
                events_status = 'success'
                print(f"{'  ' * depth} Events loaded OK")

            except Exception as e:
                events_status = f'error: {str(e)}'
                print(f"{'  ' * depth} Events load error: {str(e)}")
                post_log('load_events_mags', dttm_from, dttm_to, events_status, params=region_params)

            try:
                load_to_postgres(mags, 'magnitudes')
                mags_status = 'success'
                print(f"{'  ' * depth} Magnitudes loaded OK")

            except Exception as e:
                mags_status = f'error: {str(e)}'
                print(f"{'  ' * depth} Magnitudes load error: {str(e)}")
                post_log('load_events_mags', dttm_from, dttm_to, mags_status, params=region_params)

    # ========== ОШИБКА HTTP/ПАРСИНГА ==========

    except Exception as e:
        events_status = f'error: {str(e)}'
        mags_status = f'error: {str(e)}'

        print(f"{'  ' * depth} Error: {str(e)}")
        post_log('load_events_mags', dttm_from, dttm_to, events_status, params=region_params)

        # Если сектор пустой -- не делим его
        if str(e).startswith("no element found:"):
            return

        # ——— ДЕЛИМ РЕГИОН ———
        print(f"{'  ' * depth} Splitting region into 4 parts...")
        post_log('load_events_mags', dttm_from, dttm_to, 'will_split', params=region_params)

        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2

        subregions = [
            (lat_mid, lat_max, lon_min, lon_mid, "NW"),
            (lat_mid, lat_max, lon_mid, lon_max, "NE"),
            (lat_min, lat_mid, lon_min, lon_mid, "SW"),
            (lat_min, lat_mid, lon_mid, lon_max, "SE"),
        ]

        for lat_min_sub, lat_max_sub, lon_min_sub, lon_max_sub, direction in subregions:
            time.sleep(2)
            load_recursion_events_and_mags(
                dttm_from, dttm_to,
                lat_min_sub, lat_max_sub,
                lon_min_sub, lon_max_sub,
                min_mag=min_mag,
                depth=depth + 1,
                max_depth=max_depth
            )

    # ========== ИТОГОВЫЙ ЛОГ ==========
    summary_params = region_params.copy()
    summary_params['events_status'] = events_status
    summary_params['magnitudes_status'] = mags_status

    print(f"{'  ' * depth} {region_id} processing complete")
    post_log('load_events_mags', dttm_from, dttm_to, 'complete', params=summary_params)

# ========= Итоговая функция для приходов =========
def load_recursion_arrivals(dttm_from, dttm_to, lat_min, lat_max, lon_min, lon_max, min_mag=0.5, depth=0, max_depth=7):
    region_id = f"[lat:{lat_min:.2f}-{lat_max:.2f}, lon:{lon_min:.2f}-{lon_max:.2f}]"
    region_params = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "dttm_from": dttm_from,
        "dttm_to": dttm_to,
        "min_mag": min_mag,
        "depth": depth
    }
    
    if depth >= max_depth:
        print(f"{'  ' * depth} Max recursion depth reached for {region_id}")
        post_log('load_arrivals', dttm_from, dttm_to, 'max_depth', params=region_params)
        return
    
    
    post_log('load_arrivals', dttm_from, dttm_to, load_status='start', params=region_params)
    print(f"{'  ' * depth} Loading region {region_id}")
    
    # ========== ЗАГРУЗКА ARRIVALS ==========
    arrivals_status = None
    
    try:
        print(f"{'  ' * depth} Attempting to load...")
        arrivals = get_arrivals_by_rect(dttm_from, dttm_to, min_mag, 
                                       lat_min, lat_max, lon_min, lon_max)
        
        if arrivals.empty:
            print(f"{'  ' * depth} No data found")
            post_log('load_arrivals', dttm_from, dttm_to, 'success_no_data', params=region_params)
            arrivals_status = 'success_no_data'
        else:
            print(f"{'  ' * depth} Loading {len(arrivals)} arrivals...")
            post_log('load_arrivals', dttm_from, dttm_to, 'http_success', params=region_params)
            try:
                load_to_postgres(arrivals, 'arrivals')
                post_log('load_arrivals', dttm_from, dttm_to, 'load_success', params=region_params)
                arrivals_status = 'success'
                print(f"{'  ' * depth}Successfully loaded")
            except Exception as e:
                arrivals_status = f'error: {str(e)}'
                print(f"{'  ' * depth} Error: {str(e)}")
                post_log('load_arrivals', dttm_from, dttm_to, arrivals_status, params=region_params)
                
    except Exception as e:
        arrivals_status = f'error: {str(e)}'
        print(f"{'  ' * depth} Error: {str(e)}")
        post_log('load_arrivals', dttm_from, dttm_to, arrivals_status, params=region_params)
        
        if str(e).startswith("no element found:") is True:
            return

        print(f"{'  ' * depth} Splitting region into 4 parts...")
        post_log('load_arrivals', dttm_from, dttm_to, 'will_split', params=region_params)
        
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2
        
        subregions = [
            (lat_mid, lat_max, lon_min, lon_mid, "NW"),
            (lat_mid, lat_max, lon_mid, lon_max, "NE"),
            (lat_min, lat_mid, lon_min, lon_mid, "SW"),
            (lat_min, lat_mid, lon_mid, lon_max, "SE"),
        ]
        
        for lat_min_sub, lat_max_sub, lon_min_sub, lon_max_sub, direction in subregions:
            time.sleep(2)
            load_recursion_arrivals(dttm_from, dttm_to, 
                                       lat_min_sub, lat_max_sub, 
                                       lon_min_sub, lon_max_sub, 
                                       min_mag=min_mag, 
                                       depth=depth + 1,
                                       max_depth=max_depth)
    
    # ========== ИТОГОВЫЙ ЛОГ ==========
    summary_params = region_params.copy()
    summary_params['arrivals_status'] = arrivals_status
    
    print(f"{'  ' * depth} {region_id} processing complete")
    post_log('load_arrivals', dttm_from, dttm_to, 'complete', params=summary_params)


# load_recursion_events_and_mags(
#     '2002-01-01', 
#     '2010-12-31', 
#     48, 60, 
#     155, 167, 
#     min_mag=1, 
#     depth=0,
#     max_depth=6
# )

load_recursion_arrivals(
    '2002-01-01', 
    '2010-12-31', 
    48, 60, 
    155, 167, 
    min_mag=1, 
    depth=0,
    max_depth=7
)
