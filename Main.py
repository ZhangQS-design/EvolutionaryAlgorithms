"""
@description: 算法执行
"""
import math
import os
from concurrent import futures
import pandas as pd
import time
from AuxiliaryFunctions import initialization, evolution, get_chromosome_link_code, \
    save_excellent_gene_list, generate_attribution_for_each_node, tasks_release, \
    construct_tasks_table_and_update_tasks_pool, severd_ratio
import sys
import threading

from Configuration import task_num_range, lam

sys.setrecursionlimit(10000)


def pre_handle():
    generate_attribution_for_each_node()
    construct_tasks_table_and_update_tasks_pool()
    tasks_release()


def execute():
    # 遗传迭代次数
    inheritance_epochs = 150

    t = 1
    dispatch_pre_border_time = [1]
    total_delay_num = 0
    total_comp_delay = 0
    total_trans_delay = 0
    total_wait_delay = 0
    total_delay = 0

    while t < 300:
        # 种群初始化
        init_s_t = time.time()
        e_seeds, e_delays, e_hop_delays, parallel_execution, serial_execution, e_comp_delays, e_trans_delays, \
            e_wait_delays = initialization(dispatch_pre_time=dispatch_pre_border_time[-1], dispatch_time=t)
        # e_seeds, e_delays, e_hop_delays, parallel_execution, serial_execution = initialization(dispatch_pre_time=28,
        #                                                  dispatch_time=40)
        print("=================================================================================================")

        if e_seeds.__len__() != 0:
            init_e_t = time.time()
            print("init + repair：" + str(init_e_t - init_s_t))
            # dispatch_server_statistic.append((receive_tasks_num, server_tasks_num))
            print("初代种群---->")
            print("遗传决定时延（所有染色体的任务平均时延）：")
            print(e_delays)
            print('遗传决定时延（min）：' + str(min(e_delays)))
            print("种群hop时延（所有染色体的最大任务时延）：")
            print(e_hop_delays)
            print('被选中的hop时延：' + str(e_hop_delays[e_delays.index(min(e_delays))]))
            print("任务计算并行度")
            print(parallel_execution)
            print("任务计算串行度")
            print(serial_execution)

            evo_s_time = time.time()
            # 100次演化

            for evolution_step in range(inheritance_epochs):
                print('------------------------------------t:' + str(t) + ', e_step:' + str(evolution_step) +
                      '------------------------------------')

                e_seeds, e_delays, e_hop_delays, parallel_execution, serial_execution, \
                    e_comp_delays, e_trans_delays, e_wait_delays \
                    = evolution(t, e_seeds, e_delays, e_hop_delays, parallel_execution, serial_execution,
                                e_comp_delays, e_trans_delays, e_wait_delays)
                # print("串行度" + str(serial_execution))
                # print("并行度" + str(parallel_execution))

                # print('genetic_delay_list:' + str(e_delays))
                print('min_value: ' + str(min(e_delays)))
                # print('min_value e_comp_delays: ' + str(e_comp_delays[e_delays.index(min(e_delays))]))
                # print('min_value e_trans_delays: ' + str(e_trans_delays[e_delays.index(min(e_delays))]))
                # print('min_value: e_wait_delays: ' + str(e_wait_delays[e_delays.index(min(e_delays))]))

                print('---------------------------------------------------------------------\n')

                # print('时间'+str(t)+'下的第'+str(evolution_step)+'次演化结果编码')
                # print([get_chromosome_link_code(special_c) for special_c in e_seeds])

            evo_e_time = time.time()
            print('时间' + str(t) + '下的种群演化时间：' + str(evo_e_time - evo_s_time))

            print("*******************")
            print("进化100代后的种群：")
            print("进化后种群遗传时延：")
            print(e_delays)
            print('选择的遗传时延: ' + str(min(e_delays)))
            print('选择的计算时延: ' + str(e_comp_delays[e_delays.index(min(e_delays))]))
            print('选择的通信时延: ' + str(e_trans_delays[e_delays.index(min(e_delays))]))
            print('选择的等待时延: ' + str(e_wait_delays[e_delays.index(min(e_delays))]))
            a = e_comp_delays[e_delays.index(min(e_delays))] + e_trans_delays[e_delays.index(min(e_delays))] + \
                e_wait_delays[e_delays.index(min(e_delays))]
            print('测试: a = ', a)
            # print("进化后的种群hop时延：")
            # print(e_hop_delays)
            # print('选择的hop时延：' + str(e_hop_delays[e_delays.index(min(e_delays))]))
            print("*******************")
            # 取具有最低任务平均时延的染色体提作为当前时间下的最优选择
            better_chromosome = e_seeds[e_delays.index(min(e_delays))]
            # 保存最优染色体
            save_excellent_gene_list(time_s=t, selected_chromosome=better_chromosome)
            total_delay += min(e_delays)
            total_comp_delay += e_comp_delays[e_delays.index(min(e_delays))]
            total_trans_delay += e_trans_delays[e_delays.index(min(e_delays))]
            total_wait_delay += e_wait_delays[e_delays.index(min(e_delays))]
            total_delay_num += 1

        dispatch_pre_border_time.append(t)

        # 寻找100个染色体中计算delay最低下标的 hop作为下一个时间间隔（这里还有点问题）
        # 目前已经找到
        time_hop = math.ceil(e_hop_delays[e_delays.index(min(e_delays))])
        print(time_hop)
        # is_hop = 1
        # while is_hop:
        #     if time_hop < 10:
        #         t += 10
        #         is_hop = 0
        #     else:
        #         time_hop -= 10
        #         t += 10
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
    hop_delay = pd.DataFrame(data=dispatch_pre_border_time)
    hop_delay.to_csv('analysis/hop_delay.csv')
    parallel_execution_task_ratio = pd.DataFrame(data=parallel_execution)
    parallel_execution_task_ratio.to_csv('analysis/parallel_execution_task_ratio.csv')
    serial_execution_task_ratio = pd.DataFrame(data=serial_execution)
    serial_execution_task_ratio.to_csv('analysis/serial_execution_task_ratio.csv')
    path = os.getcwd() + "//result"
    with open(path + "//" + "task_num" + str(task_num_range[1]) + "    lam" + str(lam) + "   uav_location.txt",
              'w') as f:
        f.writelines("comp_delay, trans_delay, wait_delay, aver_delay")
        f.writelines("\n")
        f.writelines(str(total_task_aver_comp_delay) + "," + str(total_task_aver_trans_delay) + "," + str(
            total_task_aver_wait_delay) + "," + str(total_task_aver_delay))

if __name__ == '__main__':
    # generate_attribution_for_each_node()
    # construct_tasks_table_and_update_tasks_pool()
    # tasks_release()
    # pre_handle()
    execute()
    # generate_attribution_for_each_node()
    # construct_tasks_table_and_update_tasks_pool()
    # tasks_release()
    # sys.setrecursionlimit(100000)
    # threading.stack_size(200000000)
    # thread = threading.Thread(target=execute)
    # thread.start()
    # print("服务率为" + str(severd_ratio()))
