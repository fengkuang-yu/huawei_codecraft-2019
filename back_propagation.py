def generate_road_map(roadData, crossData, answer_road_path):
    """
    生成道路的车辆数据
    :param roadData:
    :return:车辆的
    """
    road_map = dict()
    road_info = dict()
    answer_map = dict()
    cross_map = dict()

    for i in range(len(roadData)):
        road_map[roadData.loc[i].id] = \
            [[[] for _ in range(roadData.loc[i].channel)] for _ in range(roadData.loc[i].isDuplex + 1)]
        road_info[roadData.loc[i].id] = list(roadData.values[i])

    for i in range(len(answer_road_path)):
        # answer_map是车辆的行驶线路map
        answer_map[answer_road_path[i][0]] = answer_road_path[i][2:]

    for i in range(len(crossData)):
        cross_map[roadData.loc[i].id] = list(roadData.values[i])[1:]

    return road_map, road_info, answer_map, cross_map

road_map, car_map, answer_map = generate_road_map(roadData, crossData, answer_road_path)
# 对于road_map来说0是正向，1是反向。

class Car:
    def __init__(self, id, v_lim, s1, length, state=0):
        self.id = id
        self.v_lim = v_lim
        self.s1 = s1
        self.state = state
        self.channel = []
        self.road_length = length

    def getChannel(self, in_road)->int:
        """
        根据当前的条件获取下一条路的流入车道
        :param cur_cross: 当前交叉路口
        :param cur_road: 当前道路
        :param dir: 车辆的转向,1表示左转，2表示直行，3表示右转
        :return: 进入的车道号
        """
        for i in range(len(in_road)):
            if in_road[i] == [] or in_road[i][-1].s1 < in_road[i][-1].length:
                return in_road[i]
        raise Exception('获取进入通道失败')

    def moveToNextRoad(self, channel, r_v_lim, next_road_length):
        self.channel.remove(self)
        self.channel = channel
        self.s1 = next_road_length - (min(self.v_lim, r_v_lim) - self.s1)
        self.state = 0
        self.road_length = next_road_length
        channel.append(self)

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

def drive_car_in_road_to_end(cur_road, channel:list, answer_map):
    if not channel:
        return
    # 先对车道内的车辆进行标记,0表示终止状态，1表示等待状态
    if channel[0].s1 < channel[0].v_lim :
        channel[0].state = 0
        channel[0].s1 -= channel[0].v_lim
    else:
        channel[0].state = 1

    for i in range(1, len(channel)):
        # 先判断车辆是否到达终点了
        if answer_map[channel[i].id][-1] == cur_road:
            channel.remove(channel[i])
        # 再判断可以到在道路中不能行驶到终点的车
        if channel[i - 1].state == 0 or channel[i].s1 - channel[i-1] < channel[i].v_lim:
            channel[i].state = 0  # 车辆到达终止状态
            channel[i].s1 = max(channel[i].v_lim, channel[i-1].s1 + 1)  # 移动车辆的位置
        else:
            channel[i].state = 1  # 车辆为等待状态
    return

def get_car_direction(car: Car, cur_cross: int, cur_road: int, cross_map: dict, answer_map: dict)->int:
    car_path_list = answer_map[car.id]
    cur_index = car_path_list.index(cur_road)
    if cur_index == len(car_path_list)-1:
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
    road_out = [1,1,0,0]
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
        if  2 not in temp_dir and 1 in temp_dir:
            dir.remove(3)
    return dir

def get_in_road(cur_cross: int, cur_road: int, dir: int, cross_map:dict, road_map: dict):
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
                if road_info[cur_road][5] == cur_cross:
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
                    car.moveToNextRoad(channel, road_info[2], road_info[1])
                    road_road_wait_car.remove(car)
                    drive_car_in_road_to_end(cur_road, channel, answer_map)

def drive_car_into_road(answer_map):
    # 将车库内的车安排上路
    ans
    pass

