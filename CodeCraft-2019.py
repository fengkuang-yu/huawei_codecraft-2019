# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   CodeCraft-2019.py
@Time    :   2019/3/16 15:56
@Desc    :
"""

from collections import defaultdict
from heapq import *

import numpy as np
import pandas as pd


def read_data(car_path, road_path, cross_path):
    """
    从给定的路径读取数据
    :param filepath: 数据路径
    :return: 读取的数据
    """

    def read_from_txt(path):
        "从txt文件中读出数据"
        with open(path) as f:
            data = []
            head = f.readline()[2: -2].split(',')
            line = f.readline()
            while line:
                line = line.strip('(').strip('\n').strip(')').split(",")
                data.append(list(map(int, line)))
                line = f.readline()
            data = pd.DataFrame(data, columns=head)
        return data

    carData = read_from_txt(car_path)
    roadData = read_from_txt(road_path)
    crossData = read_from_txt(cross_path)
    return (carData, roadData, crossData)


def create_road_between_cross_graph(roadData, crossData):
    """
    生成道路查找矩阵cross_graph;
    生成cross间权值矩阵weight_matrix;
    生成边 edges;
    :param roadData:
    :return:
    """
    M = 99999  # This represents that there is no link.
    edges = []
    cross_graph = dict()
    adjacent_matrix = np.full((len(crossData), len(crossData)), M)

    for i in range(len(roadData)):
        cross_graph['{}{}'.format(roadData.values[i, 4] - 1, roadData.values[i, 5] - 1)] = roadData.id[i]
        adjacent_matrix[roadData.values[i, 4] - 1, roadData.values[i, 5] - 1] \
            = roadData.values[i, 1]
        if roadData.isDuplex[i] == 1:
            cross_graph['{}{}'.format(roadData.values[i, 5] - 1, roadData.iloc[i, 4] - 1)] = roadData.id[i]
            adjacent_matrix[roadData.values[i, 5] - 1, roadData.values[i, 4] - 1] \
                = roadData.values[i, 1]

    # (i,j) is a link; adjacent[i, j] here is 1, the length of link (i,j).
    for i in range(len(adjacent_matrix)):
        for j in range(len(adjacent_matrix[0])):
            if i != j and adjacent_matrix[i][j] != M:
                edges.append((i, j, adjacent_matrix[i][j]))
    return cross_graph, adjacent_matrix, edges


def dijkstra(graph, start, end):
    """
    根据图的权值矩阵计算两点之间的最短路径dijkstra算法实现，
    有向图和路由的源点作为函数的输入，最短路径作为输出
    :param graph:
    :param start:
    :param end:
    :return:
    """

    def dijkstra_raw(edges, from_node, to_node):
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        while q:
            (cost, v1, path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    return cost, path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost + c, v2, path))
        return float("inf"), []

    def dijkstra(edges, from_node, to_node):
        len_shortest_path = -1
        ret_path = []
        length, path_queue = dijkstra_raw(edges, from_node, to_node)
        if len(path_queue) > 0:
            len_shortest_path = length  ## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)  ## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right) > 0:
                left = right[0]
                ret_path.append(left)  ## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path, ret_path

    return dijkstra(graph, start, end)


def update_adjacent_matrix(road):
    """
    根据现在的状态更新权值矩阵
    :return:
    """
    pass


def generate_cross_path(carData, edges):
    """
    计算路径上经过的节点
    :param carData: 车辆数据
    :return: 车辆经过的节点answer_node_path --> list(id, PlanTime, node1, node2 ...)
    """
    # 给car的数据进行排序，按照出发时间-->出发地点-->速度
    carData.sort_values(['planTime', 'from', 'speed'], ascending=[True, True, False], inplace=True)
    answer_node_path = []  # 总的输出结果

    # 先求出沿线通过的交叉路口
    for i_car in range(len(carData)):
        ans_one = []  # 给出当前车的结果路线
        _, path = dijkstra(edges, carData.iloc[i_car, 1] - 1, carData.iloc[i_car, 2] - 1)
        ans_one.append(carData.iloc[i_car, 0])
        ans_one.append(carData.iloc[i_car, 4])
        ans_one += path
        answer_node_path.append(ans_one)
    return answer_node_path


def generate_answer(answer_node_path, cross_road_map):
    """
    根据经过的交叉口生成道路id的结果
    :param answer_node_path: 规划路线中经过的交叉点的list
    :param cross_road_map: 交叉点与道路编号的映射图
    :return: 返回生成的规划线路的list
    """
    answer_road_path = []
    for i_car in range(len(answer_node_path)):
        ans_one = []
        ans_one += answer_node_path[i_car][0: 2]
        temp = answer_node_path[i_car][2:]
        if len(temp) >= 2:
            for i in range(len(temp) - 1):
                ans_one.append(cross_road_map[str(temp[i]) + str(temp[i + 1])])
        answer_road_path.append(ans_one)
    return answer_road_path


def update_departure_time(answer_road_path):
    # 建立同一时刻放入车道中的车的数量
    # 现在的answer_total为lis(list(id, route...)),调整出发时间
    carNum = 1  # 相同时刻从同一个cross出发的车辆数
    temp_from = answer_road_path[0][2]  # 当前车辆的出发cross
    count = 1
    for i_car in range(1, len(answer_road_path)):
        # 后面一辆车的出发时间和前一辆车的出发时间相同时carNum加1
        # if temp_from == answer_total[i_car][2]:
        #     carNum += 100
        #     answer_total[i_car][1] = carNum
        # else:
        #     temp_from = answer_total[i_car][2]
        #     carNum = answer_total[i_car][1]
        answer_road_path[i_car][1] = carNum
        if count % 3 == 0:
            carNum += 1
        count += 1
    return answer_road_path


def write_answer_file(answer_list, answer_path):
    """
    将list型的结果写入answer.txt文件
    :param answer_list:
    :param answer_path:
    :return:
    """
    with open(answer_path, 'w') as f:
        f.write('#(carId,StartTime,RoadID...)')
        for cur_line in answer_list:
            f.write('\n')
            f.write(str(tuple(cur_line)))
    return


# class Car:
#     def __init__(self, v_car, path):
#         self.v_car = v_car
#         self.path = path
#         self.cur_rode = path[0]
#         # 车辆的状态0表示终止 v_car <= s1，1表示等待行驶v_car > s1
#         self.state = 0
#         self.s1 = min()
#         self.s2 = min()
#
# class Road:
#     def __init__(self, id, length, v_lim, lan_nums):
#         self.id = id
#         self.length = length
#         self.v_lim = v_lim
#         self.num_lane = lan_nums
#         self.lane = []
#         for i in range(lan_nums):
#             self.lane.append([])
#
#     def allocate_car(self, car_list):
#         temp = self.lane[0].pop()
#
# def coordinate_cross(cross_road_map):
#
#     pass
#
#
# def mark_car():
#     for road in road_all:

def generate_road_map(roadData, carData, answer_road_path):
    """
    生成道路的车辆数据
    :param roadData:
    :return:车辆的
    """
    road_map = dict()
    car_map = dict()
    answer_map = dict()
    for i in range(len(roadData)):
        road_map[roadData.loc[i].id] = \
            [[[] for _ in range(roadData.loc[i].channel)] for _ in range(roadData.loc[i].isDuplex + 1)]

    for i in range(len(carData)):
        # list(state, speed, direction, s1)
        # 其中state 可选0,1
        # direction 可选0,1,2分别表示直行、左拐和右拐
        # s1表示当前路段剩余道路
        car_map[carData.loc[i].id] = [0, 0, 0, 0]

    for i in range(len(answer_road_path)):
        # answer_map是车辆的行驶线路map
        answer_map[answer_road_path[i][0]] = answer_road_path[i][2:]

    return road_map, car_map, answer_map



if __name__ == '__main__':
    answer_path = r'D:\Users\yyh\Pycharm_workspace\leetcode\answer.txt'
    road_path = r'D:\Users\yyh\Pycharm_workspace\leetcode\road.txt'
    car_path = r'D:\Users\yyh\Pycharm_workspace\leetcode\car.txt'
    cross_path = r'D:\Users\yyh\Pycharm_workspace\leetcode\cross.txt'

    carData, roadData, crossData = read_data(car_path, road_path, cross_path)
    cross_road_map, adjacent_matrix, edges = create_road_between_cross_graph(roadData, crossData)
    answer_node_path = generate_cross_path(carData, edges)
    answer_road_path = generate_answer(answer_node_path, cross_road_map)
    answer_road_path = update_departure_time(answer_road_path)
    write_answer_file(answer_road_path, answer_path)
