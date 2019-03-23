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

road_map, car_map, answer_map = generate_road_map()

class Car:
    def __init__(self, id, v_lim, s1, length, state=0):
        self.id = id
        self.v_lim = v_lim
        self.s1 = s1
        self.state = state
        self.channel = []
        self.road_length = length

    def getChannel(self, cur_cross: int, cur_road: int, dir: int, cross_map:dict, road_map: dict)->int:
        """
        根据当前的条件获取下一条路的流入车道
        :param cur_cross: 当前交叉路口
        :param cur_road: 当前道路
        :param dir: 车辆的转向,1表示左转，2表示直行，3表示右转
        :return: 进入的车道号
        """
        cross_road_list = cross_map[cur_cross]
        cur_index = cross_road_list.index(cur_road)
        in_road_index = (cur_index + dir) % 4
        in_road_number = cross_road_list[in_road_index]
        # 双向车道，判断是走正还是反
        in_road = road_map[in_road_number][0] if in_road_index < 2 else road_map[in_road_number][1]
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


def one_second(road_info, road_map, answer_map, roadData, cross_map):
    """

    :param road_info: 存放路的全部信息的dict
    :param road_map: 存放车的dict
    :param answer_map:
    :param roadData:
    :param cross_map:
    :return:
    """

    def getCarFromRoad(road: int, dir: int)->Car:
        # 判断当前路段的车辆判断顺序

        pass

    def drive_car_in_road_to_end(channel:list, wait_cars:list):
        if not channel:
            return

        # 先对车道内的车辆进行标记,0表示终止状态，1表示等待状态
        if channel[0].s1 < channel[0].v_lim:
            channel[0].state = 0
            channel[0].s1 -= channel[0].v_lim
        else:
            channel[0].state = 1
            wait_cars.append(cur_channel[0])

        for i in range(1, len(channel)):
            if channel[i - 1].state == 0 or channel[i].s1 - channel[i-1] < channel[i].v_lim:
                channel[i].state = 0  # 车辆到达终止状态
                channel[i].s1 = max(channel[i].v_lim, channel[i-1].s1 + 1)  # 移动车辆的位置
            else:
                channel[i].state = 1  # 车辆为等待状态
                wait_cars.append(cur_channel[i])
        return

    def get_car_direction(car: Car, cur_cross: int, cur_road: int, answer_map: dict, cross_map: dict)->int:
        car_path_list = answer_map[car.id]
        cur_index = car_path_list.index(cur_road)
        if cur_index == len(car_path_list)-1:
            # 进入此处表示出现错误
            print('error in "get_car_direction"')
            return 0
        in_road = car_path_list[cur_index + 1]
        start = cross_map[cur_cross].index(cur_road)
        end = cross_map[cur_cross].index(in_road)
        return (end + 4 - start) % 4

    def get_road_direction(cur_road, cur_cross):
        # 获取当前道路的优先级list()
        # 取当前每个通道的第一辆车判断优先级是否符合
        pass

    def is_conflict(cur_road, cur_cross, dir, ):
        if

        if dir == 2:
            return False
        if dir == 1:

        pass

    wait_cars = []
    for cur_road_num in road_map:
        for cur_road in road_map[cur_road_num]:
            for cur_channel in cur_road:
                drive_car_in_road_to_end(cur_channel, wait_cars)

    while wait_cars:
        for cur_cross in cross_map:
            cross_road_list = list(filter(lambda x: x != -1, cross_map[cur_cross]))
            cross_road_list.sort()
            for cur_road in cross_road_list:
                road_dir = 1
                if road_info[cur_road][5] == cur_cross:
                    road_dir = 0  # 表示路是正向的，从start -> end

                # 现在的车可以走的方向
                dir = get_road_direction(car, cur_cross, cur_road, answer_map, cross_map)
                car = getCarFromRoad(cur_road, dir)
                conflict = is_conflict()
                if conflict:
                    break
                channel = car.getChannel()
                car.moveToNextRoad(channel, r_v_lim, next_road_length)
                drive_car_in_road_to_end(cur_channel)


def driveCarIntoRoad(answer_map):
    # 将车库内的车安排上路
    pass

