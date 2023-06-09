"""
@description: experiment setting
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data/vehicles.txt', header=0, index_col=1)
data_uav = pd.read_csv('data/uav_location.txt', header=0, index_col=1)
vehicles_attr = pd.read_csv('data/vehicles_attr.csv', header=None, index_col=None)
uavs_attrs = pd.read_csv('data/uavs_attr.csv', header=None, index_col=None)

# 开放线程的数目
thread_num = 5
# 研究区域边界
x_axis_range = 1500
y_axis_range = 1500


# 车辆之间，车辆与雾节点之间的最大通信距离
v2v_limit_distance = 250
v2f_limit_distance = 375
u2vf_limit_distance = 250


# 4个雾节点的坐标
fogs_location = {
    'left_down': (100, 375, 375),
    'left_up': (101, 375, 1125),
    'right_down': (102, 1125, 375),
    'right_up': (103, 1125, 1125),
}

# 6个无人机初始坐标
uavs_location = {
    '1': (105, 275, 275),
    '2': (106, 975, 270),
    '3': (107, 275, 825),
    '4': (108, 275, 1125),
    '5': (109, 975, 675),
    '6': (110, 975, 900),
}

total_tasks_number = 300

# 云端属性配置
cloud_id = 104
# cloud_compute_delay = 0

cloud_transmission_rate = 60   # Mbps
cloud_compute_ability = 300 * pow(10, 9)
cloud_compute_delay = 0

cloud_memory = 1000*1024    # M
cloud_accept_threshold = 3

# 无人机属性配置
uav_compute_power_range = (3, 6)  # 10**9
uav_trans_rate_range = (5, 10)
uav_memory_range = (100, 150)   # M

# 车辆节点与雾节点属性配置
v_compute_power_range = (10, 15)  # 10**9
v_trans_rate_range = (10, 15)
v_memory_range = (200, 250)   # M

fog_compute_power_range = (25, 50)  # 10**9
fog_trans_rate_range = (22, 27)  # fog to vehicle
fog_memory_range = (600, 800)   # M


lam = 70

# 车辆(TaskGroup)中task数目的上限
# MAX_TASK_COUNT_IN_A_VEHICLE = 4
task_num_range = (6, 8)

# 单一task的属性配置
sub_task_num_range = (1, 1)    # 子任务数目的随机范围

# sub_task的属性配置
upload_size_range = (0.1, 5.0)    # M
download_size_range = (5, 15)   # M
compute_cost_range = (1, 20)    # 10**9
storage_cost_range = (1, 2)  # M


p_cross = 0.9   # 交叉概率
p_mutation = 0.05  # 变异概率
p_reverse = 0.5     # 翻转概率

tasks_table_path = 'data/tasks_table.csv'
tasks_pool_path = 'data/tasks_pool.data'



