import ucvm
import graphics
from ucvm import VelocityModel
from ucvm import pd
import numpy as np

import skfmm

point_1 = ucvm.Point(-118.573, 35.396, 8500) # 2012-01-25 06:51:21.700
point_2 = ucvm.Point(-118.474, 35.66278, 0) # 2012-01-25 06:51:27.220
# depth = 1000.0  

# 2018-01-28 10:53:24.200
# 2018-01-28 10:53:32.900 P
# 2018-01-28 10:53:39.850 S


# 2015-01-07 22:46:12.870
# 2015-01-07 22:46:19.620 P
# 2015-01-07 22:46:25.030 S

# point_1 = ucvm.Point(-114.0, 36.0, 0)
# point_2 = ucvm.Point(-119.0 ,32.0, 20000)

# for i in range(19):
#     P_time = ucvm.get_model_time_arrival(point_1, point_2, 'P')
#     S_time = ucvm.get_model_time_arrival(point_1, point_2, 'S')
#     print(P_time, S_time)

# grid = ucvm.get_2d_grid(point_1, point_2, 5000, step=0.001)

# model = ucvm.VelocityModel(point_1, point_2, 10001, 10001, 1001)

#---------------------------------------------------------
grid = ucvm.get_3d_grid(point_1, point_2, 101, 301, 101)

# print(grid)
vels, _ = ucvm.get_velocities(grid)


vels = vels.reshape(101, 301, 101)

phi = np.ones((101, 301, 101))
phi[0, 0, 0] = -1  # Начальная точка волны

print(vels.shape)

travel_time = skfmm.travel_time(phi, vels)

end_time = travel_time[100, 300, 100]

print(end_time*100)

#---------------------------------------------------------
# print(grid[1][95:105])

# grid = ucvm.get_ray_grid(point_1, point_2, step=0.00001)



# vp_data, vs_data = ucvm.get_velocities(grid)

# model.set_velocity(vp_data, vs_data)

# model.load_cvh()

# model.save('32-36-114-119-20000-HQ.npz')





# model = VelocityModel.load('32-36-114-119-20000-main.npz')
# print(model.ddepth, model.dlat, model.dlon)

# time_arrival = ucvm.calculate_wave_time(model.vp, model.min_lon, model.max_lon, model.nx, model.min_lat, model.max_lat, model.ny, model.min_depth, model.max_depth, model.nz, (30, 30, 10), (150, 150, 70))
# print(time_arrival)
# ray_grid = model.trace_ray(point_1, point_2, 'vp')

# vels = model.vp

# print(vels.shape)

# delta_time = model.get_time_arrival(point_1, point_2, 'vp')
# print(delta_time)

# vp, _ = model.get_velocity_at_points(ray_grid)
# print(vp)
# delta_bending = ucvm.calculate_time_arrival(ray_grid, vp)
# print(delta_bending)

# ray_grid = ucvm.get_ray_grid(point_1, point_2)
# vp_s, _ = ucvm.get_velocities(ray_grid)

# ucvm_time = ucvm.calculate_time_arrival(ray_grid, vp_s)
# print(ucvm_time)


# print(model.min_lat, model.min_lon, model.max_lat, model.max_lon)

# df = ucvm.get_arrivals('2015-03-14', '2015-07-20', model.min_lon, model.min_lat, model.max_lon, model.max_lat)

# df = ucvm.add_model_time_arrival(model, df)

# df.to_csv('arrival_diffs.csv')






# df = pd.read_csv('arrival_diffs.csv')
# print(df)

# graphics.plot_hist(df.loc[df['arrival_type'].isin(['P', 'Pg']), 'deviation'], 'hist_p.png', trimmed=True)


# point_3 = ucvm.Point(-117.4, 34.5)
# point_4 = ucvm.Point(-118.1 ,33.5)
# grid_2d = ucvm.get_2d_grid(point_3, point_4, 501, 501, 4500)

# print('here', grid_2d[1])

# vp_2d_model, vs_2d_model = model.get_velocity_at_points(grid_2d)

# vp_2d_ucvm, vs_2d_ucvm = ucvm.get_velocities(grid_2d)

# print(len(vs_2d_model), len(vs_2d_ucvm), len(grid_2d[0]))
# print(ucvm.calculate_time_arrival(grid, vp_data))
# print(ucvm.calculate_time_arrival(grid, vs_data))

# print(grid[0])

# graphics.plot_vels_2d(grid_2d, vs_2d_model, 'model_2d_main_4500.png')
# graphics.plot_vels_2d(grid_2d, vs_2d_ucvm, 'ucvm_2d_4500.png')

# vs_data = ucvm.query_vs_grid(*grid, depth)
