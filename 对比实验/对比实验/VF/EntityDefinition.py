"""
@description: 实体定义
"""
import random
from Configuration import upload_size_range, download_size_range, compute_cost_range, \
    storage_cost_range, sub_task_num_range

import numpy as np


class SubTask:
    def __init__(self, timestamp, vehicle_id, task_order, sub_task_order, selection_space, upload_size, download_size,
                 compute_cost, storage_cost):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_task_order = task_order
        self.belonging_sub_task_order = sub_task_order

        self.upload_size = upload_size
        self.download_size = download_size
        self.compute_cost = compute_cost
        self.storage_cost = storage_cost

        self.allocate_node_id = -1
        self.selection_space = selection_space

        self.up_delay = -100000
        self.wait_delay = -100000
        self.dispatch_wait_delay = -100000
        self.compute_delay = -100000
        self.punish_delay = -100000
        self.dw_delay = -100000
        self.whole_delay = -100000

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
        sub_task_info['wait_delay'] = self.wait_delay
        sub_task_info['dispatch_wait_delay'] = self.dispatch_wait_delay
        sub_task_info['compute_delay'] = self.compute_delay
        sub_task_info['punish_delay'] = self.punish_delay
        sub_task_info['dw_delay'] = self.dw_delay
        sub_task_info['whole_delay'] = self.whole_delay
        return sub_task_info


class Task:
    def __init__(self, timestamp, vehicle_id, task_order, communication_space, sub_tasks_num, task_info_dict):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_task_order = task_order
        self.t_communication_space = communication_space
        self.sub_task_num = sub_tasks_num
        self.sub_task_pool = [SubTask(timestamp, vehicle_id, task_order, sub_id, communication_space,
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

    def update_task_part_attr(self, timestamp, vehicle_id, task_order, communication_space):
        self.timestamp = timestamp
        self.belonging_vehicle_id = vehicle_id
        self.belonging_task_order = task_order
        self.t_communication_space = communication_space

        for e in self.sub_task_pool:
            e.timestamp = timestamp
            e.belonging_vehicle_id = vehicle_id
            e.belonging_task_order = task_order
            e.selection_space = communication_space


class TaskGroup:
    def __init__(self, timestamp, vehicle_id, task_pool, communication_space):
        self.timestamp = timestamp
        self.vehicle_id = vehicle_id
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
        infos['task_num'] = self.task_num
        infos['task_pool'] = {}
        for idx in range(self.task_num):
            task_info_i = self.task_pool[idx].get_task_info()
            infos['task_pool']['task'+str(idx)] = task_info_i
        return infos
