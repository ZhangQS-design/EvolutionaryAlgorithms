"""
@description: 实体定义
"""
from Configuration import cloud_id
import numpy as np
import copy
import random


class SubTask:
    def __init__(self, timestamp, vehicle_id, uav_id, task_order, sub_tasks_num, sub_task_order, selection_space, upload_size, download_size,
                 compute_cost, storage_cost):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_uav_id = uav_id
        self.belonging_task_order = task_order
        self.sub_tasks_num = sub_tasks_num
        self.belonging_sub_task_order = sub_task_order

        self.upload_size = upload_size
        self.download_size = download_size
        self.compute_cost = compute_cost
        self.storage_cost = storage_cost

        self.allocate_node_id = -1
        self.selection_space = selection_space

        self.up_delay = -100000
        self.down_delay = -10000
        self.wait_delay = -100000
        self.dispatch_wait_delay = -100000
        self.compute_delay = -100000
        self.punish_delay = -100000
        self.dw_delay = -100000
        self.whole_delay = -100000

        self.id_in_link = -100000

    def get_sub_task_info(self):
        sub_task_info = dict()
        sub_task_info['sub_task_order'] = self.belonging_sub_task_order
        sub_task_info['upload_size'] = self.upload_size
        sub_task_info['download_size'] = self.download_size
        sub_task_info['compute_cost'] = self.compute_cost
        sub_task_info['storage_cost'] = self.storage_cost
        sub_task_info['allocate_node_id'] = self.allocate_node_id
        sub_task_info['selection_space'] = self.selection_space

        sub_task_info['up_delay'] = self.up_delay
        sub_task_info['down_delay'] = self.down_delay
        sub_task_info['wait_delay'] = self.wait_delay
        sub_task_info['dispatch_wait_delay'] = self.dispatch_wait_delay
        sub_task_info['compute_delay'] = self.compute_delay
        sub_task_info['punish_delay'] = self.punish_delay
        sub_task_info['dw_delay'] = self.dw_delay
        sub_task_info['whole_delay'] = self.whole_delay
        return sub_task_info

    def delay_resetting(self):
        # 无人机-卸载节点时延
        self.up_delay = -100000
        # 卸载节点-目标车辆时延
        self.down_delay = -100000
        # 排队时延
        self.wait_delay = -100000
        self.dispatch_wait_delay = -100000
        # 计算时延
        self.compute_delay = -100000
        # 惩罚时延
        self.punish_delay = -100000
        self.dw_delay = -100000
        self.whole_delay = -100000

class Task:
    def __init__(self, timestamp, vehicle_id, uav_id, task_order, communication_space, sub_tasks_num, task_info_dict):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_uav_id = uav_id
        self.belonging_task_order = task_order
        self.t_communication_space = communication_space
        self.sub_task_num = sub_tasks_num
        self.sub_task_pool = [SubTask(timestamp, vehicle_id, uav_id,
                                      task_order, sub_tasks_num, sub_id, communication_space,
                                      task_info_dict['sub_task_'+str(sub_id)][0],
                                      task_info_dict['sub_task_' + str(sub_id)][1],
                                      task_info_dict['sub_task_' + str(sub_id)][2],
                                      task_info_dict['sub_task_' + str(sub_id)][3]
                                      ) for sub_id in range(self.sub_task_num)]

    def get_task_info(self):
        task_info = dict()
        task_info['task_order'] = self.belonging_task_order
        task_info['sub_task_num'] = self.sub_task_num
        task_info['sub_task_pool'] = {}
        for idx in range(self.sub_task_num):
            sub_info_i = self.sub_task_pool[idx].get_sub_task_info()
            task_info['sub_task_pool']['sub_task'+str(idx)] = sub_info_i
        return task_info

    def update_task_part_attr(self, timestamp, vehicle_id, uav_id, task_order, communication_space):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_uav_id = uav_id
        self.belonging_task_order = task_order
        self.t_communication_space = communication_space

        for e in self.sub_task_pool:
            e.timestamp = timestamp
            e.belonging_vehicle_id = vehicle_id
            e.belonging_uav_id = uav_id
            e.belonging_task_order = task_order
            e.selection_space = communication_space


class TaskGroup:
    def __init__(self, timestamp, vehicle_id, uav_id, task_pool, communication_space):
        self.timestamp = timestamp
        self.vehicle_id = vehicle_id
        self.uav_id = uav_id
        self.task_num = len(task_pool)
        self.task_pool = task_pool
        self.communication_space = communication_space

    def get_task_num(self):
        return self.task_num

    def get_sub_task_num(self):
        sub_task_num_list = []
        for idx in range(self.task_num):
            sub_task_num_list.append(self.task_pool[idx].sub_task_num)
        return sub_task_num_list

    def get_sub_task_objects(self):
        sub_objects = []
        for idx in range(self.task_num):
            sub_objects.extend(self.task_pool[idx].sub_task_pool)
        return sub_objects

    def get_all_information(self):
        infos = dict()
        infos['timestamp'] = self.timestamp
        infos['vehicle_id'] = self.vehicle_id
        infos['uav_id'] = self.uav_id
        infos['task_num'] = self.task_num
        infos['task_pool'] = {}
        for idx in range(self.task_num):
            task_info_i = self.task_pool[idx].get_task_info()
            infos['task_pool']['task'+str(idx)] = task_info_i
        return infos


# 可选调度的节点中心
class SourceCenter:
    def __init__(self, available_memory):
        self.reset_memory = copy.deepcopy(available_memory)
        self.available_memory = available_memory
        self.request_dicts = dict()
        # for x in range(0, 105):
        # +UAV
        for x in range(0, 111):
            self.request_dicts[x] = dict()

    def init_malloc(self, request_id, request_size, reachable_indices, test_threshold):
        test_threshold = test_threshold
        while True:
            # test_threshold 含义？
            init_malloc_node_id = random.choice(reachable_indices)
            if init_malloc_node_id == 104 and test_threshold > 0:
                test_threshold -= 1
                continue
            else:
                self.request_dicts[init_malloc_node_id][request_id] = request_size
                self.available_memory[init_malloc_node_id] -= request_size
                break
        return init_malloc_node_id

    def repair_malloc(self, request_id, request_size, reachable_indices):
        gene_space_v = [x for x in reachable_indices if 0 <= x <= 99]
        gene_space_f = [x for x in reachable_indices if 100 <= x <= 103]
        # 添加可以卸载的无人机
        gene_space_uav = [x for x in reachable_indices if 105 <= x <= 110]

        malloc_flag = False
        malloc_node_id = cloud_id

        if malloc_flag is False:
            for n_id in gene_space_v:
                if self.available_memory[n_id] >= request_size:
                    malloc_node_id = n_id
                    self.request_dicts[malloc_node_id][request_id] = request_size
                    self.available_memory[malloc_node_id] -= request_size
                    malloc_flag = True
                    break

        if malloc_flag is False:
            for n_id in gene_space_f:
                if self.available_memory[n_id] >= request_size:
                    malloc_node_id = n_id
                    self.request_dicts[malloc_node_id][request_id] = request_size
                    self.available_memory[malloc_node_id] -= request_size
                    malloc_flag = True
                    break

        # 添加无人机的选项
        if malloc_flag is False:
            for n_id in gene_space_uav:
                if self.available_memory[n_id] >= request_size:
                    malloc_node_id = n_id
                    self.request_dicts[malloc_node_id][request_id] = request_size
                    self.available_memory[malloc_node_id] -= request_size
                    malloc_flag = True
                    break

        if malloc_flag is False:
            self.request_dicts[malloc_node_id][request_id] = request_size

        return malloc_node_id

    def repair_malloc1(self, request_id, request_size):
        malloc_node_id = cloud_id
        self.request_dicts[malloc_node_id][request_id] = request_size
        return malloc_node_id

    def free_and_assign_malloc(self, request_id, request_size, old_node_id, assign_node_id):
        if request_id not in self.request_dicts[old_node_id].keys():
            print('old_node_id:' + str(old_node_id))
            print(request_id, self.request_dicts[old_node_id])

        del self.request_dicts[old_node_id][request_id]
        self.available_memory[old_node_id] += request_size
        self.request_dicts[assign_node_id][request_id] = request_size
        self.available_memory -= request_size

    def reset(self):
        self.available_memory = copy.deepcopy(self.reset_memory)
        self.request_dicts = dict()
        # +UAV
        for x in range(0, 111):
        # for x in range(0, 105):
            self.request_dicts[x] = dict()

    # 修复存储大小不够的节点
    def self_detect_repair(self):
        redistribution_request_ids = []
        # for idx in range(105):
        for idx in range(111):
            if self.available_memory[idx] < 0:
                self.available_memory[idx] = self.reset_memory[idx]
                for pair in sorted(self.request_dicts[idx], key=self.request_dicts[idx].__getitem__, reverse=True):
                    if self.request_dicts[idx][pair] <= self.available_memory[idx]:
                        # 修改
                        # self.available_memory -= self.request_dicts[idx][pair]
                        self.available_memory[idx] -= self.request_dicts[idx][pair]
                    else:
                        redistribution_request_ids.append(pair)
                        del self.request_dicts[idx][pair]
        return redistribution_request_ids


# s = SourceCenter(np.ones(105)*100)
# for i in range(1, 5):
#     if i == 4:
#         i = i * 2
#     s.malloc(i-1, i*10+i, [e for e in range(0, 105)])
# print(s.request_dicts)
# s.free_and_assign_malloc(7, 88, 1, 0)
# print(s.request_dicts)
#
# print(s.self_detect_repair())
# print(s.request_dicts)


