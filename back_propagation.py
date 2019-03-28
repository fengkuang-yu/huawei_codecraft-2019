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


class Car:
    def __init__(self, id, v_lim, s1, length, state=0):
        self.id = id
        self.v_lim = v_lim
        self.s1 = s1
        self.state = state
        self.channel = []
        self.road_length = length

    def getChannel(self, in_road) -> int:
        """
        根据当前的条件获取下一条路的流入车道
        :param cur_cross: 当前交叉路口
        :param cur_road: 当前道路
        :param dir: 车辆的转向,1表示左转，2表示直行，3表示右转
        :return: 进入的车道号
        """
        for i in range(len(in_road)):
            if in_road[i] == [] or in_road[i][-1].s1 == in_road[i][-1].length - 1:
                return in_road[i]
        raise Exception('获取进入通道失败')

    def moveToNextRoad(self, channel, r_v_lim, next_road_length):
        self.channel.remove(self)
        self.channel = channel
        self.v_lim = min(self.v_lim, r_v_lim)
        self.s1 = next_road_length - (self.v_lim - self.s1)
        self.road_length = next_road_length
        self.state = 0
        channel.append(self)


def generate_road_map(roadData, crossData, carData, answer_road_path):
    """
    生成道路的车辆数据
    :param roadData:
    :return:车辆的
    """
    road_map = dict()
    road_info = dict()
    answer_map = dict()
    cross_map = dict()
    car_map = dict()

    for i in range(len(roadData)):
        road_map[roadData.loc[i].id] = \
            [[[] for _ in range(roadData.loc[i].channel)] for _ in range(roadData.loc[i].isDuplex + 1)]
        road_info[roadData.loc[i].id] = list(roadData.values[i])[1:]

    for i in range(len(answer_road_path)):
        # answer_map是车辆的行驶线路map
        answer_map[answer_road_path[i][0]] = answer_road_path[i][2:]

    for i in range(len(crossData)):
        cross_map[roadData.loc[i].id] = list(roadData.values[i])[1:]

    for i in range(len(carData)):
        car_map[carData.loc[i].id] = list(carData.values[i])[1:]

    return road_map, road_info, answer_map, cross_map, car_map


def get_car_from_road(cur_road, cur_cross, road_map: dict, cross_map: dict, answer_map: dict, dir: list):
    # 判断当前路段的车辆判断顺序
    road_dir = 0 if cross_map[cur_road].index(cur_road) // 2 else 1
    temp_road = road_map[cur_road][road_dir]
    temp_car = []
    for temp_channel in temp_road:
        temp_car += temp_channel[:1]
    for dir_order in dir:
        for car_order in temp_car:
            if dir_order == get_car_direction(car_order, cur_cross, cur_road, cross_map, answer_map):
                return dir_order, car_order


def drive_car_in_road_to_end(cur_road, channel: list, answer_map):
    if not channel:
        return
    # 先对车道内的车辆进行标记,0表示终止状态，1表示等待状态
    if channel[0].s1 < channel[0].v_lim:
        channel[0].state = 0
        channel[0].s1 -= channel[0].v_lim
    else:
        channel[0].state = 1

    for i in range(1, len(channel)):
        # 先判断车辆是否到达终点了
        if answer_map[channel[i].id][-1] == cur_road:
            channel.remove(channel[i])
        # 再判断可以到在道路中不能行驶到终点的车
        if channel[i - 1].state == 0 or channel[i].s1 - channel[i - 1] < channel[i].v_lim:
            channel[i].state = 0  # 车辆到达终止状态
            channel[i].s1 = max(channel[i].v_lim, channel[i - 1].s1 + 1)  # 移动车辆的位置
        else:
            channel[i].state = 1  # 车辆为等待状态
    return


def get_car_direction(car: Car, cur_cross: int, cur_road: int, cross_map: dict, answer_map: dict) -> int:
    car_path_list = answer_map[car.id]
    cur_index = car_path_list.index(cur_road)
    if cur_index == len(car_path_list) - 1:
        # 进入此处表示出现错误
        print('判断车辆是否到达终点的程序出错')
        return 0
    in_road = car_path_list[cur_index + 1]
    start = cross_map[cur_cross].index(cur_road)
    end = cross_map[cur_cross].index(in_road)
    return (end + 4 - start) % 4


def get_road_direction(cur_cross: int, cur_road: int, cross_map: dict, road_map: dict, answer_map: dict) -> list:
    # 获取当前道路的优先级list()
    # 取当前每个通道的第一辆车判断优先级是否符合
    dir = []
    road_out = [1, 1, 0, 0]
    cur_roads_list = cross_map[cur_cross]
    cur_index = cur_roads_list.index(cur_road)
    left_index = (cur_index + 1) % 4
    direct_index = (cur_index + 2) % 4
    right_index = (cur_index + 3) % 4
    # 判断是否可以直行
    dir.append(2)
    if cur_roads_list[direct_index] == -1:
        dir.remove(2)

    # 判断是否可以左转
    dir.append(1)
    if cur_roads_list[right_index] != -1:
        channel_top_car = []
        for temp_channel in road_map[cur_roads_list[right_index]][road_out[right_index]]:
            channel_top_car += temp_channel[:1]  # 取出每个车道的最前面一辆车判断是否发生冲突
        for temp_car in channel_top_car:
            if temp_car.state == 1:
                if get_car_direction(temp_car, cur_cross, cur_road, cross_map, answer_map) == 2:
                    dir.remove(1)
                    break

    # 判断是否可以右转
    dir.append(3)
    if cur_roads_list[left_index] != -1:
        channel_top_car = []
        for temp_channel in road_map[cur_roads_list[left_index]][road_out[left_index]]:
            channel_top_car += temp_channel[:1]  # 取出每个车道的最前面一辆车判断是否发生冲突
        for temp_car in channel_top_car:
            if temp_car.state == 1:
                if get_car_direction(temp_car, cur_cross, cur_road, cross_map, answer_map) == 2:
                    dir.remove(3)
                    break
    if cur_roads_list[direct_index] != -1 and (3 in dir):
        channel_top_car = []
        temp_dir = []
        for temp_channel in road_map[cur_roads_list[direct_index]][road_out[direct_index]]:
            channel_top_car += temp_channel[:1]  # 取出每个车道的最前面一辆车判断是否发生冲突
        for temp_car in channel_top_car:
            if temp_car.state == 1:
                temp_dir.append(get_car_direction(temp_car, cur_cross, cur_road, cross_map, answer_map))
        if 2 not in temp_dir and 1 in temp_dir:
            dir.remove(3)
    return dir


def get_in_road(cur_cross: int, cur_road: int, dir: int, cross_map: dict, road_map: dict):
    cross_road_list = cross_map[cur_cross]
    cur_index = cross_road_list.index(cur_road)
    in_road_index = (cur_index + dir) % 4
    in_road_number = cross_road_list[in_road_index]
    # 双向车道，判断是走正还是反
    in_road = road_map[in_road_number][0] if in_road_index < 2 else road_map[in_road_number][1]
    return in_road


def one_second(road_info, road_map, answer_map, cross_map):
    wait_cars = []
    for cur_road_num in road_map:
        for cur_road in road_map[cur_road_num]:
            for cur_channel in cur_road:
                drive_car_in_road_to_end(cur_road, cur_channel, answer_map)

    while wait_cars:
        for cur_cross in cross_map:
            cross_road_list = list(filter(lambda x: x != -1, cross_map[cur_cross]))
            cross_road_list.sort()
            for cur_road in cross_road_list:
                road_dir = 1
                if road_info[cur_road][4] == cur_cross:
                    road_dir = 0  # 表示路是正向的，从start -> end
                # 现在的车可以走的方向
                road_road_wait_car = []
                for cur_channel in road_road_wait_car:
                    for cur_car in cur_channel:
                        if cur_car.state == 1:
                            road_road_wait_car.append(cur_car)
                dir_list = get_road_direction(cur_cross, cur_road, cross_map, road_map, answer_map)
                while road_road_wait_car:
                    dir, car = get_car_from_road(cur_road, cur_cross, road_map, cross_map, answer_map, dir_list)
                    in_road = get_in_road(cur_cross, cur_road, dir, cross_map, road_map)
                    if in_road[-1] and in_road[-1][-1].s1 == in_road[-1][-1].length - 1:
                        break
                    channel = car.getChannel(in_road)
                    car.moveToNextRoad(channel, road_info[1], road_info[0])
                    road_road_wait_car.remove(car)
                    drive_car_in_road_to_end(cur_road, channel, answer_map)


def drive_car_into_road(cur_time_step, car_map, answer_map, road_info, cross_map, answer_road_path):
    # 找到这个时间出发的车放在cars_ready中
    i = 0
    for i in range(len(answer_road_path)):
        if answer_road_path[i][1] != cur_time_step:
            break
        else:
            i += 1
    cars_ready, answer_road_path = answer_road_path[:i], answer_road_path[i:]

    # 按照出发路段和车辆id进行排序上路，不能上路的放在delay_cars中
    cars_ready.sort(key=lambda x: (x[2], x[0]))
    delay_cars = []
    for temp_car in cars_ready:
        car_id = temp_car[0]
        start_road = temp_car[2]
        len_road = road_info[start_road][0]
        v_lim = min(road_info[start_road][1], car_map[car_id][2])
        cross_id = car_map[car_id][0]
        PorN = cross_map[cross_id].index(start_road) // 2

        in_road = start_road[PorN]
        # 如果不能安排上车
        if in_road[-1] and in_road[-1][-1].s1 == in_road[-1][-1].road_length - 1:
            temp_car[1] += 1
            answer_map[car_id][0] += 1
            delay_cars.append(temp_car)
        else:
            for i in range(len(in_road)):
                if in_road[i] == [] or in_road[i][-1].s1 > len_road - v_lim:
                    car = Car(car_id, v_lim, len_road - v_lim, len_road)
                    in_road[i].append(car)
                    break
                elif in_road[i][-1].s1 != len_road - 1:
                    car = Car(car_id, v_lim, in_road[i][-1].s1 + 1, len_road)
                    in_road[i].append(car)
                    break
    return delay_cars + answer_road_path, answer_map


def check_road(road_map):
    res_car = []
    for temp_road in road_map:
        for i in range(len(temp_road)):
            for j in temp_road[i]:
                if i == 0:
                    res = 'Forward id:'
                else:
                    res = 'Reverse id:'
                res += str(temp_road[i][j].id) + ' s1:'
                res += str(temp_road[i][j].road_length - temp_road[i][j].s1)
                res_car.append(res)
    return res_car


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
    road_map, road_info, answer_map, cross_map, car_map = generate_road_map(roadData, crossData, carData,
                                                                            answer_road_path)
