"""
@description: 算法执行
"""
import math
import os
import time
from AuxiliaryFunctions import initialization, get_chromosome_link_code, \
    save_excellent_gene_list, severd_ratio, generate_attribution_for_each_node
import sys
import pandas as pd

from Configuration import task_num_range, lam

sys.setrecursionlimit(10000)

# construct_tasks_table_and_update_tasks_pool()
#
# # 是否为节点生成属性：包括 计算能力，计算时延，存储等配置
# generate_attribution_for_each_node()
# 预先发布所有时间下的任务，确定研究场景
# tasks_release()

# 遗传迭代次数
inheritance_epochs = 100

t = 1
dispatch_pre_border_time = [1]
total_delay_num = 0
total_comp_delay = 0
total_trans_delay = 0
total_wait_delay = 0
total_delay = 0
# dispatch_server_statistic = []

# generate_attribution_for_each_node()
# exit()
# print(end="1")

while t < 300:
    # 种群初始化
    init_s_t = time.time()
    e_seeds, e_delays, e_hop_delays, e_comp_delays, e_trans_delays, e_wait_delays = \
        initialization(dispatch_pre_time=dispatch_pre_border_time[-1], dispatch_time=t)
    # for kk in e_seeds:
    #     print("时间" + str(t) + "下放置的节点为" + str(kk.allocate_node_id))
    if e_seeds.__len__() != 0:
        init_e_t = time.time()
        print("贪心染色体构造时间：" + str(init_e_t-init_s_t))
        # dispatch_server_statistic.append((receive_tasks_num, server_tasks_num))
        print("贪心染色体的任务平均时延：")
        print(e_delays)

        print("贪心染色体的最大任务时延：")
        print(e_hop_delays)

        # 保存贪心染色体
        save_excellent_gene_list(time_s=t, selected_chromosome=e_seeds)
        total_delay += e_delays[0]
        total_comp_delay += e_comp_delays[0]
        total_trans_delay += e_trans_delays[0]
        total_wait_delay += e_wait_delays[0]
        total_delay_num += 1

    dispatch_pre_border_time.append(t+1)

    time_hop = e_hop_delays[0]

    # t = t + math.ceil(time_hop)
    t = t + 10
    print("下一次调度时间：" + str(t))

total_task_aver_comp_delay = total_comp_delay / total_delay_num
print("total_task_aver_comp_delay", total_task_aver_comp_delay)

total_task_aver_trans_delay = total_trans_delay / total_delay_num
print("total_task_aver_trans_delay", total_task_aver_trans_delay)

total_task_aver_wait_delay = total_wait_delay / total_delay_num
print("total_task_aver_wait_delay", total_task_aver_wait_delay)

total_task_aver_delay = total_delay / total_delay_num
print("total_task_aver_delay", total_task_aver_delay)
print("服务率为" + str(severd_ratio()))

path = os.getcwd() + "//result"
with open(path + "//" + "task_num" + str(task_num_range[1]) + "    lam" + str(lam) + "   uav_location.txt", 'w') as f:
    f.writelines("comp_delay, trans_delay, wait_delay, aver_delay")
    f.writelines("\n")
    f.writelines(str(total_task_aver_comp_delay) + "," + str(total_task_aver_trans_delay) + "," + str(total_task_aver_wait_delay)
                 + "," + str(total_task_aver_delay))
hop_delay = pd.DataFrame(data=dispatch_pre_border_time)
hop_delay.to_csv('analysis/hop_delay.csv')











































