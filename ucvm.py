import numpy as np
import pandas as pd
import subprocess
import tempfile
import psycopg2
import skfmm
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta
from pyproj import Geod

os.environ["LD_LIBRARY_PATH"] = ":".join([
    "/mnt/disk01/egor/ucvm_final/lib",
    "/mnt/disk01/egor/ucvm_final/lib/proj/lib",
    os.environ.get("LD_LIBRARY_PATH", "")
])

db_params = {
    'dbname': 'gis',
    'user': 'gis',
    'password': '123456',
    'host': '10.0.62.59',
    'port': '55432' 
}

class Point:
    def __init__(self, lon, lat, depth=0):
        self.lon = lon
        self.lat = lat
        self.depth = depth


class VelocityModel:
    def __init__(self, point1, point2, nx, ny, nz):
        self.point_1 = point1
        self.point_2 = point2
        self.min_lon = min(point1.lon, point2.lon)
        self.max_lon = max(point1.lon, point2.lon)
        self.min_lat = min(point1.lat, point2.lat)
        self.max_lat = max(point1.lat, point2.lat)
        self.min_depth = min(point1.depth, point2.depth)
        self.max_depth = max(point1.depth, point2.depth)

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dlon = (self.max_lon - self.min_lon) / (nx - 1)
        self.dlat = (self.max_lat - self.min_lat) / (ny - 1)
        self.ddepth = (self.max_depth - self.min_depth) / (nz - 1)

        self.vp = np.zeros((nx, ny, nz), dtype=np.float32)
        self.vs = np.zeros((nx, ny, nz), dtype=np.float32)

    def set_velocity(self, vp_flat, vs_flat):
        expected_size = self.nx * self.ny * self.nz
        assert len(vp_flat) == expected_size, f"vp size {len(vp_flat)} != expected {expected_size}"
        assert len(vs_flat) == expected_size, f"vs size {len(vs_flat)} != expected {expected_size}"

        self.vp = np.reshape(vp_flat, (self.nx, self.ny, self.nz)).astype(np.float32)
        self.vs = np.reshape(vs_flat, (self.nx, self.ny, self.nz)).astype(np.float32)


    def _get_indices_and_weights(self, val, min_val, delta, n):

        f = (val - min_val) / delta
        
        i0 = int(np.floor(f))
        i1 = i0 + 1
        w1 = f - i0
        w0 = 1.0 - w1
        if i0 < 0 or i1 >= n:
            return None, None, None 
        return i0, i1, (w0, w1)

    def _is_inside(self, lon, lat, depth):
        return (self.min_lon <= lon <= self.max_lon and
                self.min_lat <= lat <= self.max_lat and
                self.min_depth <= depth <= self.max_depth)

    def _interp_velocity(self, lon, lat, depth, grid):
        i = int((lon - self.min_lon) / self.dlon)
        j = int((lat - self.min_lat) / self.dlat)
        k = int((depth - self.min_depth) / self.ddepth)

        if (0 <= i < self.nx) and (0 <= j < self.ny) and (0 <= k < self.nz):
            return grid[i, j, k]
        return 0.0

    def _segment_distance(self, lon1, lat1, dep1, lon2, lat2, dep2):
        # Псевдоевклидово расстояние в 3D
        from math import sqrt
        horiz = haversine(lon1, lat1, lon2, lat2)
        vert = abs(dep2 - dep1)
        return sqrt(horiz**2 + vert**2)

    def get_velocity_at_points(self, grid):
        lons = grid[0]
        lats = grid[1]
        depths = grid[2]

        result_vp = np.zeros_like(lats, dtype=np.float32)
        result_vs = np.zeros_like(lats, dtype=np.float32)

        for idx in range(len(lats)):
            # print(lats, idx, lats[idx])
            lat, lon, depth = lats[idx], lons[idx], depths[idx]
            # print(lat)

            ix0, ix1, wx = self._get_indices_and_weights(lon, self.min_lon, self.dlon, self.nx)
            iy0, iy1, wy = self._get_indices_and_weights(lat, self.min_lat, self.dlat, self.ny)
            iz0, iz1, wz = self._get_indices_and_weights(depth, self.min_depth, self.ddepth, self.nz)

            if None in (ix0, iy0, iz0):
                continue  # вне сетки

            vp_val = 0.0
            vs_val = 0.0
            for dx, wx_val in zip([ix0, ix1], wx):
                for dy, wy_val in zip([iy0, iy1], wy):
                    for dz, wz_val in zip([iz0, iz1], wz):
                        weight = wx_val * wy_val * wz_val
                        vp_val += self.vp[dx, dy, dz] * weight
                        vs_val += self.vs[dx, dy, dz] * weight

            result_vp[idx] = vp_val
            result_vs[idx] = vs_val

        return result_vp, result_vs
    
    def load_cvh(self):
        grid = get_3d_grid(self.point_1, self.point_2, self.nx, self.ny, self.nz)
        vp, vs = get_velocities(grid)
        self.set_velocity(vp, vs)

    def save(self, filename):
        np.savez_compressed(filename,
            vp=self.vp,
            vs=self.vs,
            min_lon=self.min_lon,
            max_lon=self.max_lon,
            min_lat=self.min_lat,
            max_lat=self.max_lat,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz
        )

    def get_time_arrival(self, point1: Point, point2: Point, wave_type, step=0.001):
        ray_grid = get_ray_grid(point1, point2, step)
        vp, vs = self.get_velocity_at_points(ray_grid)
        if wave_type == 'vp':
            velocities = vp
        elif wave_type == 'vs':
            velocities = vs
        else:
            raise ValueError("wave_type must be 'vp' or 'vs'")

        return calculate_time_arrival(ray_grid, velocities)

    def trace_ray(self, source: Point, receiver: Point, wave_type='vp'):
        velocity = getattr(self, wave_type)
        print('here')

        x = np.linspace(self.min_lon, self.max_lon, self.nx)
        y = np.linspace(self.min_lat, self.max_lat, self.ny)
        z = np.linspace(self.min_depth, self.max_depth, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        phi = np.ones_like(velocity)
        ix = int(round((source.lon - self.min_lon) / self.dlon))
        iy = int(round((source.lat - self.min_lat) / self.dlat))
        iz = int(round((source.depth - self.min_depth) / self.ddepth))
        phi[ix, iy, iz] = -1
        print('here')
        METERS_PER_DEG_LAT = 111_000
        METERS_PER_DEG_LON = 111_000 * np.cos(np.radians((self.min_lat + self.max_lat) / 2))

        dx = [self.dlon * METERS_PER_DEG_LON, 
        self.dlat * METERS_PER_DEG_LAT, 
        self.ddepth]

        print('here')
        travel_time = skfmm.travel_time(phi, 1.0 / velocity, dx=dx)

        print(travel_time)
        lon, lat, depth = receiver.lon, receiver.lat, receiver.depth
        lons, lats, depths = [], [], []
        print('here')
        for _ in range(1000):
            lons.append(lon)
            lats.append(lat)
            depths.append(depth)

            if not (self.min_lon <= lon <= self.max_lon and
                    self.min_lat <= lat <= self.max_lat and
                    self.min_depth <= depth <= self.max_depth):
                break

            i = (lon - self.min_lon) / self.dlon
            j = (lat - self.min_lat) / self.dlat
            k = (depth - self.min_depth) / self.ddepth

            ii = int(np.clip(i, 1, self.nx - 2))
            jj = int(np.clip(j, 1, self.ny - 2))
            kk = int(np.clip(k, 1, self.nz - 2))

            grad_x = (travel_time[ii+1, jj, kk] - travel_time[ii-1, jj, kk]) / (2 * self.dlon)
            grad_y = (travel_time[ii, jj+1, kk] - travel_time[ii, jj-1, kk]) / (2 * self.dlat)
            grad_z = (travel_time[ii, jj, kk+1] - travel_time[ii, jj, kk-1]) / (2 * self.ddepth)

            grad = np.array([grad_x, grad_y, grad_z])
            norm = np.linalg.norm(grad)

            if norm == 0:
                break

            step = -0.5
            delta = step * grad / norm

            lon += delta[0]
            lat += delta[1]
            depth += delta[2]

            if np.linalg.norm([lon - source.lon, lat - source.lat, depth - source.depth]) < max(self.dlon, self.dlat, self.ddepth):
                lons.append(source.lon)
                lats.append(source.lat)
                depths.append(source.depth)
                break

        return [np.array(lons), np.array(lats), np.array(depths)]


    @classmethod
    def load(cls, filename):
        data = np.load(filename)

        nx = int(data['nx'])
        ny = int(data['ny'])
        nz = int(data['nz'])

        min_lon = float(data['min_lon'])
        max_lon = float(data['max_lon'])
        min_lat = float(data['min_lat'])
        max_lat = float(data['max_lat'])
        min_depth = float(data['min_depth'])
        max_depth = float(data['max_depth'])

        point1 = Point(min_lon, min_lat, min_depth)
        point2 = Point(max_lon, max_lat, max_depth)

        model = cls(point1, point2, nx, ny, nz)
        model.vp = data['vp']
        model.vs = data['vs']
        return model
    

def get_2d_grid(point_1: Point, point_2: Point, nx, ny, depth):
    lon_min, lon_max = sorted([point_1.lon, point_2.lon])
    lat_min, lat_max = sorted([point_1.lat, point_2.lat])

    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)

    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
    dep_grid = np.full_like(lon_grid, depth, dtype=np.float32)

    return [
        lon_grid.ravel(),  # shape: (nx * ny,)
        lat_grid.ravel(),  # shape: (nx * ny,)
        dep_grid.ravel()   # shape: (nx * ny,)
    ]

def get_3d_grid(point_1: Point, point_2: Point, nx, ny, nz):
    lon_min, lon_max = sorted([point_1.lon, point_2.lon])
    lat_min, lat_max = sorted([point_1.lat, point_2.lat])
    dep_min, dep_max = sorted([point_1.depth, point_2.depth])

    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    deps = np.linspace(dep_min, dep_max, nz)

    lon_grid, lat_grid, dep_grid = np.meshgrid(lons, lats, deps, indexing='ij')

    grid = [
        lon_grid.ravel(),  # shape: (N,)
        lat_grid.ravel(),  # shape: (N,)
        dep_grid.ravel()   # shape: (N,)
    ]
    return grid


def get_ray_grid(point_1: Point, point_2: Point, step=0.001):
    vector = np.array([point_2.lon - point_1.lon,
                       point_2.lat - point_1.lat,
                       point_2.depth - point_1.depth])
    
    distance = np.linalg.norm(vector)

    if distance == 0:
        return np.array([[point_1.lon], [point_1.lat], [point_1.depth]])
    
    num_steps = int(np.ceil(1 / step))
    t_values = np.linspace(0, 1, num_steps + 1)
    
    lons = point_1.lon + t_values * vector[0]
    lats = point_1.lat + t_values * vector[1]
    depths = point_1.depth + t_values * vector[2]
    grid = np.array([lons, lats, depths])    
    return grid


def get_velocities(grid):
    lons, lats, depths = grid[0], grid[1], grid[2]
    vs_values = np.zeros_like(lons)
    vp_values = np.zeros_like(lons)
    
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        for lon, lat, depth in zip(lons.ravel(), lats.ravel(), depths.ravel()):
            f.write(f"{lon} {lat} {depth}\n")
        f.flush()
        
        cmd = [
            "/mnt/disk01/egor/ucvm_final/bin/ucvm_query",
            "-f", "/mnt/disk01/egor/ucvm_final/conf/ucvm.conf",
            "-m", "cvmh",
            "<", f.name
        ]
        
        result = subprocess.run(
            ' '.join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for i, line in enumerate(result.stdout.splitlines()):
            parts = line.split()
            vp_values.flat[i] = float(parts[6])
            vs_values.flat[i] = float(parts[7])
            
    return vp_values.reshape(lons.shape), vs_values.reshape(lons.shape)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a)) 
    return 6371000 * c

def calculate_time_arrival(grid, velocities):
    lons, lats, depths = grid
    time_arrival = np.zeros_like(lons)
    
    for i in range(1, len(lons)):
        
        horiz_dist = haversine(lons[i-1], lats[i-1], lons[i], lats[i])
        vert_dist = abs(depths[i] - depths[i-1])
        segment_distance = np.sqrt(horiz_dist**2 + vert_dist**2)
        avg_velocity = (velocities[i-1] + velocities[i]) / 2
        segment_time = segment_distance / avg_velocity
        
        time_arrival[i] = time_arrival[i-1] + segment_time

    result_time = timedelta(seconds=time_arrival[-1])
    
    return result_time

def get_arrivals(from_dt, to_dt, min_lon=-120.5, min_lat=31.0, max_lon=-113.5, max_lat=36.5):
    conn = psycopg2.connect(**db_params)
    query = f"""
        WITH station_locs AS (
            SELECT 
                s.station_nm, 
                s.network_nm,
                s.loc
            FROM stations s
            WHERE s.loc IS NOT NULL
        )
        SELECT 
            ST_X(e.loc) as event_lon,
            ST_Y(e.loc) as event_lat,
            e.depth as event_depth,
            a.arrival_type,
            a.arrival_dttm - e.event_dttm as actual_time_diff,
            ST_X(st.loc) as station_lon,
            ST_Y(st.loc) as station_lat
        FROM events e
        JOIN arrivals a ON e.event_id = a.event_id
        JOIN station_locs st ON a.station_nm = st.station_nm AND a.network_nm = st.network_nm
        WHERE e.loc IS NOT NULL AND a.arrival_dttm IS NOT NULL AND e.event_dttm::date >= '{from_dt}' AND e.event_dttm::date <= '{to_dt}'
        and e.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326) 
        and st.loc && ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326)
        and e.depth > 0
        """
    df = pd.read_sql(query, conn)
    df = df.loc[df['arrival_type'].isin(['P', 'S', 'Pg', 'Sg'])]

    return df

def get_model_time_arrival(point_1:Point, point_2:Point, arrival_type):
    grid = get_ray_grid(point_1, point_2, 0.1)
    vp, vs = get_velocities(grid)

    if arrival_type == 'P':
        time_diif = calculate_time_arrival(grid, vp)
    elif arrival_type == 'S':
        time_diif = calculate_time_arrival(grid, vs)
    
    return time_diif


def add_model_time_arrival(model, df):
    model_times = []

    for _, row in df.iterrows():
        # Создаем точки
        event_point = Point(row['event_lon'], row['event_lat'], row['event_depth'])
        station_point = Point(row['station_lon'], row['station_lat'], 0.0)

        # Определяем тип волны
        wave_type = 'vp' if row['arrival_type'].upper() == 'P' else 'vs'

        # Считаем время прибытия по модели
        try:
            arrival_time = model.get_time_arrival(event_point, station_point, wave_type=wave_type)
        except Exception:
            arrival_time = timedelta(seconds=0)  # или np.nan

        model_times.append(arrival_time)

    df = df.copy()
    df['model_time_arrival'] = model_times
    df['deviation'] = df['model_time_arrival'].dt.total_seconds()  - df['actual_time_diff'].dt.total_seconds() 
    return df

def geographic_to_local_coords(lon0, lat0, lons, lats):
    """
    Переводит массив lon/lat в локальные координаты (метры) относительно (lon0, lat0).
    """
    geod = Geod(ellps='WGS84')
    
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    # Создаём массивы начальных точек той же формы
    lon0_arr = np.full_like(lons, lon0)
    lat0_arr = np.full_like(lats, lat0)

    # Восточное направление (dx)
    _, _, distances_e = geod.inv(lon0_arr, lats, lons, lats)
    distances_e *= np.sign(lons - lon0)

    # Северное направление (dy)
    _, _, distances_n = geod.inv(lons, lat0_arr, lons, lats)
    distances_n *= np.sign(lats - lat0)

    return distances_e, distances_n



def calculate_wave_time(
    velocity_field,
    min_lon, max_lon, n_lon,
    min_lat, max_lat, n_lat,
    min_depth, max_depth, n_depth,
    start_indices,
    end_indices,
):
    
    # Проверки входных данных
    assert max_lon > min_lon, "Longitude range is zero!"
    assert max_lat > min_lat, "Latitude range is zero!"
    assert max_depth > min_depth, "Depth range is zero!"
    assert n_lon > 0 and n_lat > 0 and n_depth > 0, "Node counts must be > 0!"
    
    # Проверка индексов
    def check_indices(shape, indices):
        i, j, k = indices
        assert 0 <= i < shape[0], f"i (lon) index {i} is out of bounds!"
        assert 0 <= j < shape[1], f"j (lat) index {j} is out of bounds!"
        assert 0 <= k < shape[2], f"k (depth) index {k} is out of bounds!"
    
    check_indices(velocity_field.shape, start_indices)
    check_indices(velocity_field.shape, end_indices)

    # Заменяем нулевые/отрицательные скорости
    velocity_field = np.maximum(velocity_field, 1e-10)

    # Вычисляем шаги сетки в метрах
    dlon_deg = (max_lon - min_lon) / n_lon
    dlat_deg = (max_lat - min_lat) / n_lat
    ddepth_m = (max_depth - min_depth) / n_depth

    mean_lat_rad = np.radians((min_lat + max_lat) / 2)
    dlon_m = dlon_deg * 111320 * np.cos(mean_lat_rad)
    dlat_m = dlat_deg * 111320
    
    print("Start indices:", start_indices)
    print("End indices:", end_indices)
    print("Velocity at start:", velocity_field[start_indices])
    print("Grid steps (m):", dlon_m, dlat_m, ddepth_m)
    # Создаем маску
    mask = np.zeros_like(velocity_field, dtype=np.int32)
    mask[start_indices] = -1

    # Вычисляем время
    travel_time = skfmm.travel_time(
        mask,
        speed=velocity_field,
        dx=[dlon_m, dlat_m, ddepth_m]
    )

    return travel_time[end_indices]