"""
@description: 实验结果分析
"""
import os
import re
import pickle
import pandas as pd
import numpy as np

result_path = 'result'


def get_file_list(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        return files


all_chromosomes_file_name = get_file_list(result_path)

print(all_chromosomes_file_name)

chromosomes_delays_table = np.zeros((len(all_chromosomes_file_name), 7))

nodes_load_table = np.zeros((len(all_chromosomes_file_name), 5))


for file_name_idx in range(len(all_chromosomes_file_name)):

    file_name = all_chromosomes_file_name[file_name_idx]

    chromosome_time = float(re.findall(r'\d', file_name)[0])
    chromosomes_delays_table[file_name_idx, 0] = chromosome_time
    nodes_load_table[file_name_idx, 0] = chromosome_time

    one_best_chromosome_path = result_path + '/' + file_name

    with open(one_best_chromosome_path, 'rb') as f:
        one_best_chromosome = pickle.load(f)    # 这里获取了一条保存的染色体

    chromosome_ave_up_delay = 0
    chromosome_ave_wait_delay = 0
    chromosome_ave_dispatch_wait_delay = 0
    chromosome_ave_compute_delay = 0
    chromosome_ave_punish_delay = 0
    chromosome_ave_whole_delay = 0

    vehicle_load_count = 0
    fog_load_count = 0
    cloud_load_count = 0
    total_load_count = 0

    for e in one_best_chromosome:
        chromosome_ave_up_delay += e.up_delay
        chromosome_ave_wait_delay += e.wait_delay
        chromosome_ave_dispatch_wait_delay += e.dispatch_wait_delay
        chromosome_ave_compute_delay += e.compute_delay
        chromosome_ave_punish_delay += e.punish_delay
        chromosome_ave_whole_delay += e.whole_delay

        if 0 <= e.allocate_node_id <= 99:
            vehicle_load_count += 1
        elif 100 <= e.allocate_node_id <= 103:
            fog_load_count += 1
        elif e.allocate_node_id == 104:
            cloud_load_count += 1

        total_load_count += 1

    chromosome_ave_up_delay /= len(one_best_chromosome)
    chromosome_ave_wait_delay /= len(one_best_chromosome)
    chromosome_ave_dispatch_wait_delay /= len(one_best_chromosome)
    chromosome_ave_compute_delay /= len(one_best_chromosome)
    chromosome_ave_punish_delay /= len(one_best_chromosome)
    chromosome_ave_whole_delay /= len(one_best_chromosome)

    chromosomes_delays_table[file_name_idx, 1] = chromosome_ave_up_delay
    chromosomes_delays_table[file_name_idx, 2] = chromosome_ave_wait_delay
    chromosomes_delays_table[file_name_idx, 3] = chromosome_ave_dispatch_wait_delay
    chromosomes_delays_table[file_name_idx, 4] = chromosome_ave_compute_delay
    chromosomes_delays_table[file_name_idx, 5] = chromosome_ave_punish_delay
    chromosomes_delays_table[file_name_idx, 6] = chromosome_ave_whole_delay

    nodes_load_table[file_name_idx, 1] = vehicle_load_count
    nodes_load_table[file_name_idx, 2] = fog_load_count
    nodes_load_table[file_name_idx, 3] = cloud_load_count
    nodes_load_table[file_name_idx, 4] = total_load_count

chromosomes_delays_table_df = \
    pd.DataFrame(chromosomes_delays_table, index=None, columns=['time', 'ave_up_delay', 'ave_wait_delay',
                                                                'ave_dispatch_wait_delay', 'ave_compute_delay',
                                                                'ave_punish_delay', 'ave_whole_delay'])
chromosomes_delays_table_df.to_csv('analysis/chromosomes_delays_table.csv', index=False)


nodes_load_table_df = pd.DataFrame(nodes_load_table, index=None, columns=['time', 'vehicles_load', 'fogs_load',
                                                                          'cloud_load', 'total_load'])

nodes_load_table_df.to_csv('analysis/nodes_load_table.csv', index=False)










