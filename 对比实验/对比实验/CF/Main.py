"""
@description: 算法执行
"""
import math
import time
from AuxiliaryFunctions import initialization, get_chromosome_link_code, \
    save_excellent_gene_list, severd_ratio
import sys
import pandas as pd
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

# dispatch_server_statistic = []

while t < 300:
    # 种群初始化
    init_s_t = time.time()
    e_seeds, e_delays, e_hop_delays = initialization(dispatch_pre_time=dispatch_pre_border_time[-1], dispatch_time=t)
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

    dispatch_pre_border_time.append(t+1)

    time_hop = e_hop_delays[0]

    t = t + math.ceil(time_hop)
    # hop = math.ceil(time_hop)
    # schedule_interval = 10
    # is_hop = 1
    # while is_hop:
    #     if hop < schedule_interval:
    #         t = t + 10
    #         is_hop = 0
    #     else:
    #         t = t + schedule_interval
    #         hop -= schedule_interval
    print("下一次调度时间：" + str(t))

print("服务率为" + str(severd_ratio()))

hop_delay = pd.DataFrame(data=dispatch_pre_border_time)
hop_delay.to_csv('analysis/hop_delay.csv')











































