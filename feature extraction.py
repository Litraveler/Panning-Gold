import os
import sys
import csv
from time import timezone

import pandas as pd
import numpy as np
import statistics
import math

from scipy.stats import sem, kurtosis, gmean, hmean, skew
from decimal import Decimal, getcontext

def output_path(path):
    new_path_array = path.split("/")
    new_path_array[-2] = "features"
    new_path_array.pop()
    new_path_array.pop()
    new_path_array.append("features.csv")
    new_path_array = "/".join(new_path_array)

    return new_path_array

def angle_between_vectors(p0, p1, p2):
    #计算向量
    v1 = (p0[0] - p1[0], p0[1] - p1[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    # 计算点积和模长
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("两个点相同，无法计算角度")
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    if cos_theta < -1 or cos_theta > 1:
        cos_theta = min(1, max(-1, cos_theta))  # 限制值在[-1, 1]范围内
    # 计算角度
    angle = math.acos(cos_theta)
    return math.degrees(angle)  # 将弧度转换为度

def perpendicular_distance(point, start, end):
    # 计算向量
    if start == point or (end == point) or (start == end):
        return 0
    v1 = (start[0] - end[0], start[1] - end[1])
    v2 = (point[0] - start[0], point[1] - start[1])
    # 计算点积
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    if math.sqrt(v1[0] ** 2 + v1[1] ** 2) == 0:
        return math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    # 计算距离
    proj_distance = abs(dot_product / math.sqrt(v1[0] ** 2 + v1[1] ** 2))
    distance = v2[0] ** 2 + v2[1] ** 2 - proj_distance ** 2
    return distance

def median_of_three(a, b, c):
    # 先将三个数转换为列表
    numbers = [a, b, c]
    # 对列表进行排序
    numbers.sort()
    # 如果列表长度为1，返回唯一的数
    if len(numbers) == 1:
        return numbers[0]
    # 否则，返回中间的数
    else:
        return numbers[1]

def calculate_curvature_features(touch_points):
    if len(touch_points) < 3:
        raise ValueError("少于三个点，至少需要三个点：")  # 至少需要三个点
    moving_curvatures = []
    curvature_distances = []
    # 遍历三个连续的点
    for i in range(len(touch_points) - 2):
        p0, p1, p2 = touch_points[i], touch_points[i + 1], touch_points[i + 2]
        if p0 == p1 or p1 == p2:
            continue
        angle = angle_between_vectors(p0, p1, p2)
        moving_curvatures.append(angle)
        distance = perpendicular_distance(p1, p0, p2)
        curvature_distances.append(distance)
    # 计算角度平均值
    angles_rad = np.deg2rad(moving_curvatures)
    deviations_rad = np.mod(angles_rad + np.pi, 2 * np.pi) - np.pi
    # 计算平均角度
    average_moving_curvature = np.rad2deg(np.mean(deviations_rad))
    # 计算距离平均值；已经考虑了绝对值
    average_curvature_distance = np.mean(curvature_distances)
    return average_moving_curvature, average_curvature_distance

def calculate_statistics(data, time, max_deviation_index, touch_points_X, touch_points_Y):
    # 检查数据是否为空
    if len(data) == 0:
        return "数据序列为空"
    # 计算各种统计量
    #均值
    if len(data) - 1 < max_deviation_index:
        max_deviation_index = len(data) - 1
    mean = np.mean(data)
    #最大偏离点前的均值
    tvx = 0
    for i, t in enumerate(data):
        if i < max_deviation_index:
            tvx = tvx + t
    mean_before_max_deviation_point = tvx / (maxAbs_index + 1)
    #最大偏离点后的均值
    tvx = 0
    for i, t in enumerate(data):
        if i > max_deviation_index or i == max_deviation_index:
            tvx = tvx + t
    mean_after_max_deviation_point = tvx / len(data) - maxAbs_index
    #极小值
    minimum = np.min(data)
    #极大值
    maximum = np.max(data)
    minimum_index = data.index(minimum)
    maximum_index = data.index(maximum)
    maximum_value_portion = maximum_index + 1 / len(touch_points_X)
    minimum_value_portion = minimum_index + 1 / len(touch_points_X)
    time_to_reach_maximum_point = time[maximum_index] - time[0]
    time_to_reach_minimum_point = time[minimum_index] - time[0]
    maximum_point_to_end_time = time[-1] - time[maximum_index]
    minimum_point_to_end_time = time[-1] - time[minimum_index]
    length_to_reach_maximum_point = math.sqrt((touch_points_X[maximum_index] - touch_points_X[0]) ** 2 + (touch_points_Y[maximum_index] - touch_points_Y[0]) ** 2)
    length_to_reach_minimum_point = math.sqrt((touch_points_X[minimum_index] - touch_points_X[0]) ** 2 + (touch_points_Y[minimum_index] - touch_points_Y[0]) ** 2)
    length_of_Maximum_point_to_end_point = math.sqrt((touch_points_X[maximum_index] - touch_points_X[-1]) ** 2 + (touch_points_Y[maximum_index] - touch_points_Y[-1]) ** 2)
    length_of_Minimum_point_to_end_point = math.sqrt((touch_points_X[minimum_index] - touch_points_X[-1]) ** 2 + (touch_points_Y[minimum_index] - touch_points_Y[-1]) ** 2)
    #标准差
    std_dev = np.std(data, ddof=1)
    #方差
    variance = np.var(data, ddof=1)
    #变异系数
    if(mean == 0):
        cv = 0
    else:
        cv = (std_dev / mean) * 100
    #标准误差
    sem = std_dev / np.sqrt(len(data))
    #中位数
    median = np.median(data)
    #百分比位数
    twenty_percent = np.percentile(data, 20)
    lower_quartile = np.percentile(data, 25)
    second_quartile = np.percentile(data, 50)  # 中位数
    third_quartile = np.percentile(data, 75)
    eighty_percent = np.percentile(data, 80)
    interquartile_range = third_quartile - lower_quartile
    #最大偏离点的值
    largest_deviation_point_value = data[max_deviation_index]
    #二次均值
    quadratic_mean = np.sqrt(np.mean(np.square(data)))  # 二次均值
    non_zero_data = [x for x in data if x != 0]
    #调和平均数
    harmonic_mean = hmean(np.abs(non_zero_data))
    #几何平均数
    geometric_mean = gmean(np.abs(non_zero_data))
    #平均绝对偏差是一个非参数的离散度量，它对异常值不敏感，比标准差更稳健的离散度量。
    mean_abs_dev = np.mean(np.abs(data - mean))
    median_abs_dev = np.median(np.abs(data - mean))
    # 计算峰度
    kurt = calculate_kurtosis(data)
    skewness = skew(data)
    sort_first_3_points = sorted([data[0], data[1], data[2]])
    median_value_at_first_3_points = sort_first_3_points[1]
    sort_last_3_points = sorted([data[-1], data[-2], data[-3]])
    median_value_at_last_3_points = sort_last_3_points[1]
    sort_first_5_points = sorted([data[0], data[1], data[2], data[3], data[4]])
    median_value_at_first_5_points = sort_first_5_points[1]
    sort_last_5_points = sorted([data[-1], data[-2], data[-3], data[-4], data[-5]])
    median_value_at_last_5_points = sort_last_5_points[1]

    return {
        "Kurtosis": kurt,
        "Mean": mean,
        "Mean Before Max Deviation Point": mean_before_max_deviation_point,
        "Mean After Max Deviation Point": mean_after_max_deviation_point,
        "Minimum": minimum,
        "Maximum": maximum,
        "Standard Deviation": std_dev,
        "Median": median,
        "Lower Quartile": lower_quartile,
        "Second Quartile": second_quartile,
        "Third Quartile": third_quartile,
        "Quadratic Mean": quadratic_mean,
        "Harmonic Mean": harmonic_mean,
        "Geometric Mean": geometric_mean,
        "Mean Absolute Deviation": mean_abs_dev,
        "First Point value": data[0],
        "Last Point value": data[-1],
        "Twenty Percent": twenty_percent,
        "Eighty Percent": eighty_percent,
        "Largest Deviation Point Value": largest_deviation_point_value,
        "Time to reach Maximum Value": time_to_reach_maximum_point,
        "Time to reach Minimum Value": time_to_reach_minimum_point,
        "Length to reach Maximum Value": length_to_reach_maximum_point,
        "Length to reach Minimum Value": length_to_reach_minimum_point,
        "Length of Maximum Value to End": length_of_Maximum_point_to_end_point,
        "Length of Minimum Value to End": length_of_Minimum_point_to_end_point,
        "Skewness": skewness,
        "Variance": variance,
        "Coefcient of Variation": cv,
        "Standard Error of the Mean": sem,
        "Interquartile Range": interquartile_range,
        "Median Absolute Deviation": median_abs_dev,
        "Median Value at First 3 Points": median_value_at_first_3_points,
        "Median Value at First 5 Points": median_value_at_first_5_points,
        "Median Value at Last 3 Points": median_value_at_last_3_points,
        "Median Value at Last 5 Points": median_value_at_last_5_points,
        "Maximum Value Point to End Point Time": maximum_point_to_end_time,
        "Minimum Value Point to End Point Time": minimum_point_to_end_time,
        "maximum_value_portion": maximum_value_portion,
        "minimum_value_portion": minimum_value_portion,
    }

def calculate_kurtosis(data):
    if len(data) == 0:
        return 0
    high_precision_sequence = [Decimal(str(x)) for x in data]
    #不能使用sum，浮点类型高精度数据会损失精度
    mean = sum(high_precision_sequence, high_precision_sequence[0]) / len(high_precision_sequence)
    # 计算偏差的四次方和与平方和
    m4 = sum((x - mean) ** 4 for x in high_precision_sequence)
    m2 = sum((x - mean) ** 2 for x in high_precision_sequence)
    # 计算峰度（未进行Fisher-Pearson标准化）;若分子或分母为0,则意味着计算无意义，所有的点很接近
    if (m4 == 0) or (m2 ** 2 == 0):
        kurt = Decimal('0')
    else:
        kurt = m4 / (m2 ** 2)
    return kurt

def angle_mean_absolute_deviation(angles_deg):
    # 将角度转换为弧度
    angles_rad = np.radians(angles_deg)
    # 计算平均角度（弧度）
    mean_angle_rad = np.mean(angles_rad)
    # 初始化绝对偏差列表
    abs_deviations = []
    # 计算每个角度与平均角度的差，并调整到-π到π范围内
    for angle_rad in angles_rad:
        # 计算差
        diff_rad = angle_rad - mean_angle_rad
        # 调整到-π到π范围内
        adjusted_diff_rad = np.mod(diff_rad + np.pi, 2 * np.pi) - np.pi
        # 取绝对值并添加到列表中
        abs_deviations.append(np.abs(adjusted_diff_rad))
        # 计算平均绝对偏差（弧度）
    mean_abs_deviation_rad = np.mean(abs_deviations)
    # 如果需要，将平均绝对偏差转换回度
    mean_abs_deviation_deg = np.degrees(mean_abs_deviation_rad)
    return mean_abs_deviation_deg
#计算速度,时间的索引范围为0-t-1
#若data的索引范围为0-t，计算得速度列表的索引范围为0-t-1,则flag = 0
#若data的索引范围为0-t-1,计算得速度列表的索引范围为0-t-2,则flag = 1
#若data的索引范围为0-t-1,计算得速度列表的索引范围为0-t-1,则flag = 2
#若data的索引范围为0-t-2,计算得速度列表的索引范围为0-t-3,则flag = 3
def calculate_velocity(data, time, flag):
    data_velocity = []
    if flag == 0 or flag == 1:
        temData = data[0]
        for i, t in enumerate(data):
            if i != 0 and time[i - 1] != 0:
                data_velocity.append((t - temData) / time[i - 1])
                temData = t
            elif time[i - 1] == 0:
                data_velocity.append(data_velocity[i - 1])
    elif flag == 2:
        for i, t in enumerate(data):
            if time[i] != 0:
                data_velocity.append(t/time[i])
            else:
                data_velocity.append(data_velocity[i-1])
    elif flag == 3:
        temData = data[0]
        for i, t in enumerate(data):
            if i != 0 and time[i] != 0:
                data_velocity.append((t - temData) / time[i])
                temData = t
            elif time[i] == 0:
                data_velocity.append(data_velocity[i - 1])
    return data_velocity
#计算加速度,时间的索引范围为0-t-1
#若data的索引范围为0-t-2,计算得速度列表的索引范围为0-t-3,则flag = 0
#若data的索引范围为0-t-1，计算得加速度列表的索引范围为0-t-2,则flag = 1
#若data的索引范围为0-t-3，计算得加速度列表的索引范围为0-t-4,则flag = 2
def calculate_acc(data, time, flag):
    data_acc = []
    if flag == 0 or flag == 1:
        temData = data[0]
        for i, t in enumerate(data):
            if i != 0 and time[i - 1] != 0:
                data_acc.append((t - temData) / time[i - 1])
                temData = t
            elif time[i - 1] == 0:
                data_acc.append(data_acc[i - 1])
    elif flag == 2:
        temData = data[0]
        for i, t in enumerate(data):
            if i != 0 and time[i] != 0:
                data_acc.append((t - temData) / time[i])
                temData = t
            elif time[i] == 0:
                data_acc.append(data_acc[i - 1])
    return data_acc

if __name__ == "__main__":
    sorted_folder = "data_files"

    userdata = pd.read_csv("tables/userdata.csv")

    cuuid = ""
    total = len(userdata.index)
    current = 0

    #os.walk() 是 Python 的 os 模块中的一个函数，它递归遍历一个目录，返回一个三元组 (subdir, dirs, files)
    #subdir 是当前正在遍历的目录路径。
    #dirs 是该目录中的子目录列表。
    #files 是该目录中的文件列表。
    #os.walk() 函数本身就是设计来递归遍历一个目录及其所有子目录的。当你调用 os.walk() 时，它会为每个目录生成一个三元组 (subdir, dirs, files)，并且自动地遍历所有层级的子目录。
    #每次迭代会返回一个目录，然后你可以对这个目录下的子目录进行再次迭代。这样递归地进行，可以遍历所有子目录。
    for subdir, dirs, files in os.walk(sorted_folder):
        for file in files:
            #path = os.path.join(subdir, file) 组合成每个文件的完整路径。
            path = os.path.join(subdir, file)
            if path.split("/")[-1] == ".DS_Store":
                continue
            export_path = output_path(path)
            uuid = path.split("/")[-6]
            direction = path.split("/")[-4]
            if uuid != cuuid:
                cuuid = uuid
                current += 1
                sys.stdout.write("\rUser (%d/%d)" % (current, total))
                sys.stdout.flush()

            if path.split("/")[-2] != "touch_data":
                continue

            #loc是Pandas DataFrame对象的一个属性，用于通过标签索引选择数据。这里用于选择满足特定条件的行。
            #["phone_model"]这指定了从筛选出的行中选择"phone_model"列。将Pandas DataFrame中的选定列转换为NumPy数组。
            #获取UUID对应的用户的手机型号
            model = userdata.loc[userdata["uuid"] == uuid]["phone_model"].values[0]
            #-6:uudi ; model:phone_model  -5:measurement_id ; -4:gametype ; -3:iteration ; swipe_counter:swipe_counter
            output = [
                [
                    "uuid",
                    "phone_model",
                    "measurement_id",
                    "gametype",
                    "iteration",
                    "swipe_counter",
                    "discard_swipe_counter",
                    "dir_flag",
                    "dirEndToEnd",
                    "startX",
                    "startY",
                    "stopX",
                    "stopY",
                    "duration",
                    "starting_time",
                    "ending_time",
                    "length_of_trajectory",
                    "displacement",
                    "ratio",
                    "velocity_of_trajectory",
                    "velocity_of_displacement",
                    "start_to_25_point_direction",
                    "start_to_50_point_direction",
                    "start_to_75_point_direction",
                    "the_25_point_to_end_direction",
                    "the_50_point_to_end_direction",
                    "the_75_point_to_end_direction",
                    "start_to_largest_deviation_point_direction",
                    "largest_deviation_point_to_end_direction",
                    "time_to_reach_25_point",
                    "time_to_reach_50_point",
                    "time_to_reach_75_point",
                    "the_25_point_to_end_time",
                    "the_50_point_to_end_time",
                    "the_75_point_to_end_time",
                    "length_to_reach_25_point",
                    "length_to_reach_50_point",
                    "length_to_reach_75_point",
                    "the_25_point_to_end_length",
                    "the_50_point_to_end_length",
                    "the_75_point_to_end_length",
                    "average_moving_curvature",
                    "average_curvature_distance",
                    "kurtosis_phase_angle_sequences",
                    "mean_phase_angle_sequences",
                    "mean_before_max_deviation_point_phase_angle_sequences",
                    "mean_after_max_deviation_point_phase_angle_sequences",
                    "minimum_phase_angle_sequences",
                    "maximum_phase_angle_sequences",
                    "standard_deviation_phase_angle_sequences",
                    "median_phase_angle_sequences",
                    "low_quartile_phase_angle_sequences",
                    "second_quartile_phase_angle_sequences",
                    "third_quartile_phase_angle_sequences",
                    "quadratic_mean_phase_angle_sequences",
                    "harmonic_mean_phase_angle_sequences",
                    "geometric_mean_phase_angle_sequences",
                    "mean_absolute_deviation_phase_angle_sequences",
                    "first_point_value_phase_angle_sequences",
                    "last_point_value_phase_angle_sequences",
                    "twenty_percent_phase_angle_sequences",
                    "eighty_percent_phase_angle_sequences",
                    "largest_deviation_point_value_phase_angle_sequences",
                    "time_to_reach_maximum_value_phase_angle_sequences",
                    "time_to_reach_minimum_value_phase_angle_sequences",
                    "length_to_reach_maximum_value_phase_angle_sequences",
                    "length_to_reach_minimum_value_phase_angle_sequences",
                    "length_of_maximum_value_to_end_phase_angle_sequences",
                    "length_of_minimum_value_to_end_phase_angle_sequences",
                    "skewness_phase_angle_sequences",
                    "variance_phase_angle_sequences",
                    "coefcient_of_variation_phase_angle_sequences",
                    "standard_error_of_the_mean_phase_angle_sequences",
                    "interquartile_range_phase_angle_sequences",
                    "median_absolute_deviation_phase_angle_sequences",
                    "median_value_at_first_3_points_phase_angle_sequences",
                    "median_value_at_first_5_points_phase_angle_sequences",
                    "median_value_at_last_3_points_phase_angle_sequences",
                    "median_value_at_last_5_points_phase_angle_sequences",
                    "maximum_value_point_to_end_point_time_phase_angle_sequences",
                    "minimum_value_point_to_end_point_time_phase_angle_sequences",
                    "maximum_value_portion_phase_angle_sequences",
                    "minimum_value_portion_phase_angle_sequences",
                    "kurtosis_phase_angle_velocity_sequences",
                    "mean_phase_angle_velocity_sequences",
                    "mean_before_max_deviation_point_phase_angle_velocity_sequences",
                    "mean_after_max_deviation_point_phase_angle_velocity_sequences",
                    "minimum_phase_angle_velocity_sequences",
                    "maximum_phase_angle_velocity_sequences",
                    "standard_deviation_phase_angle_velocity_sequences",
                    "median_phase_angle_velocity_sequences",
                    "low_quartile_phase_angle_velocity_sequences",
                    "second_quartile_phase_angle_velocity_sequences",
                    "third_quartile_phase_angle_velocity_sequences",
                    "quadratic_mean_phase_angle_velocity_sequences",
                    "harmonic_mean_phase_angle_velocity_sequences",
                    "geometric_mean_phase_angle_velocity_sequences",
                    "mean_absolute_deviation_phase_angle_velocity_sequences",
                    "first_point_value_phase_angle_velocity_sequences",
                    "last_point_value_phase_angle_velocity_sequences",
                    "twenty_percent_phase_angle_velocity_sequences",
                    "eighty_percent_phase_angle_velocity_sequences",
                    "largest_deviation_point_value_phase_angle_velocity_sequences",
                    "time_to_reach_maximum_value_phase_angle_velocity_sequences",
                    "time_to_reach_minimum_value_phase_angle_velocity_sequences",
                    "length_to_reach_maximum_value_phase_angle_velocity_sequences",
                    "length_to_reach_minimum_value_phase_angle_velocity_sequences",
                    "length_of_maximum_value_to_end_phase_angle_velocity_sequences",
                    "length_of_minimum_value_to_end_phase_angle_velocity_sequences",
                    "skewness_phase_angle_velocity_sequences",
                    "variance_phase_angle_velocity_sequences",
                    "coefcient_of_variation_phase_angle_velocity_sequences",
                    "standard_error_of_the_mean_phase_angle_velocity_sequences",
                    "interquartile_range_phase_angle_velocity_sequences",
                    "median_absolute_deviation_phase_angle_velocity_sequences",
                    "median_value_at_first_3_points_phase_angle_velocity_sequences",
                    "median_value_at_first_5_points_phase_angle_velocity_sequences",
                    "median_value_at_last_3_points_phase_angle_velocity_sequences",
                    "median_value_at_last_5_points_phase_angle_velocity_sequences",
                    "maximum_value_point_to_end_point_time_phase_angle_velocity_sequences",
                    "minimum_value_point_to_end_point_time_phase_angle_velocity_sequences",
                    "maximum_value_portion_phase_angle_velocity_sequences",
                    "minimum_value_portion_phase_angle_velocity_sequences",
                    "kurtosis_phase_angle_acc_sequences",
                    "mean_phase_angle_acc_sequences",
                    "mean_before_max_deviation_point_phase_angle_acc_sequences",
                    "mean_after_max_deviation_point_phase_angle_acc_sequences",
                    "minimum_phase_angle_acc_sequences",
                    "maximum_phase_angle_acc_sequences",
                    "standard_deviation_phase_angle_acc_sequences",
                    "median_phase_angle_acc_sequences",
                    "low_quartile_phase_angle_acc_sequences",
                    "second_quartile_phase_angle_acc_sequences",
                    "third_quartile_phase_angle_acc_sequences",
                    "quadratic_mean_phase_angle_acc_sequences",
                    "harmonic_mean_phase_angle_acc_sequences",
                    "geometric_mean_phase_angle_acc_sequences",
                    "mean_absolute_deviation_phase_angle_acc_sequences",
                    "first_point_value_phase_angle_acc_sequences",
                    "last_point_value_phase_angle_acc_sequences",
                    "twenty_percent_phase_angle_acc_sequences",
                    "eighty_percent_phase_angle_acc_sequences",
                    "largest_deviation_point_value_phase_angle_acc_sequences",
                    "time_to_reach_maximum_value_phase_angle_acc_sequences",
                    "time_to_reach_minimum_value_phase_angle_acc_sequences",
                    "length_to_reach_maximum_value_phase_angle_acc_sequences",
                    "length_to_reach_minimum_value_phase_angle_acc_sequences",
                    "length_of_maximum_value_to_end_phase_angle_acc_sequences",
                    "length_of_minimum_value_to_end_phase_angle_acc_sequences",
                    "skewness_phase_angle_acc_sequences",
                    "variance_phase_angle_acc_sequences",
                    "coefcient_of_variation_phase_angle_acc_sequences",
                    "standard_error_of_the_mean_phase_angle_acc_sequences",
                    "interquartile_range_phase_angle_acc_sequences",
                    "median_absolute_deviation_phase_angle_acc_sequences",
                    "median_value_at_first_3_points_phase_angle_acc_sequences",
                    "median_value_at_first_5_points_phase_angle_acc_sequences",
                    "median_value_at_last_3_points_phase_angle_acc_sequences",
                    "median_value_at_last_5_points_phase_angle_acc_sequences",
                    "maximum_value_point_to_end_point_time_phase_angle_acc_sequences",
                    "minimum_value_point_to_end_point_time_phase_angle_acc_sequences",
                    "maximum_value_portion_phase_angle_acc_sequences",
                    "minimum_value_portion_phase_angle_acc_sequences",
                    "kurtosis_phase_distance_sequences",
                    "mean_phase_distance_sequences",
                    "mean_before_max_deviation_point_phase_distance_sequences",
                    "mean_after_max_deviation_point_phase_distance_sequences",
                    "minimum_phase_distance_sequences",
                    "maximum_phase_distance_sequences",
                    "standard_deviation_phase_distance_sequences",
                    "median_phase_distance_sequences",
                    "low_quartile_phase_distance_sequences",
                    "second_quartile_phase_distance_sequences",
                    "third_quartile_phase_distance_sequences",
                    "quadratic_mean_phase_distance_sequences",
                    "harmonic_mean_phase_distance_sequences",
                    "geometric_mean_phase_distance_sequences",
                    "mean_absolute_deviation_phase_distance_sequences",
                    "first_point_value_phase_distance_sequences",
                    "last_point_value_phase_distance_sequences",
                    "twenty_percent_phase_distance_sequences",
                    "eighty_percent_phase_distance_sequences",
                    "largest_deviation_point_value_phase_distance_sequences",
                    "time_to_reach_maximum_value_phase_distance_sequences",
                    "time_to_reach_minimum_value_phase_distance_sequences",
                    "length_to_reach_maximum_value_phase_distance_sequences",
                    "length_to_reach_minimum_value_phase_distance_sequences",
                    "length_of_maximum_value_to_end_phase_distance_sequences",
                    "length_of_minimum_value_to_end_phase_distance_sequences",
                    "skewness_phase_distance_sequences",
                    "variance_phase_distance_sequences",
                    "coefcient_of_variation_phase_distance_sequences",
                    "standard_error_of_the_mean_phase_distance_sequences",
                    "interquartile_range_phase_distance_sequences",
                    "median_absolute_deviation_phase_distance_sequences",
                    "median_value_at_first_3_points_phase_distance_sequences",
                    "median_value_at_first_5_points_phase_distance_sequences",
                    "median_value_at_last_3_points_phase_distance_sequences",
                    "median_value_at_last_5_points_phase_distance_sequences",
                    "maximum_value_point_to_end_point_time_phase_distance_sequences",
                    "minimum_value_point_to_end_point_time_phase_distance_sequences",
                    "maximum_value_portion_phase_distance_sequences",
                    "minimum_value_portion_phase_distance_sequences",
                    "kurtosis_phase_distance_velocity_sequences",
                    "mean_phase_distance_velocity_sequences",
                    "mean_before_max_deviation_point_phase_distance_velocity_sequences",
                    "mean_after_max_deviation_point_phase_distance_velocity_sequences",
                    "minimum_phase_distance_velocity_sequences",
                    "maximum_phase_distance_velocity_sequences",
                    "standard_deviation_phase_distance_velocity_sequences",
                    "median_phase_distance_velocity_sequences",
                    "low_quartile_phase_distance_velocity_sequences",
                    "second_quartile_phase_distance_velocity_sequences",
                    "third_quartile_phase_distance_velocity_sequences",
                    "quadratic_mean_phase_distance_velocity_sequences",
                    "harmonic_mean_phase_distance_velocity_sequences",
                    "geometric_mean_phase_distance_velocity_sequences",
                    "mean_absolute_deviation_phase_distance_velocity_sequences",
                    "first_point_value_phase_distance_velocity_sequences",
                    "last_point_value_phase_distance_velocity_sequences",
                    "twenty_percent_phase_distance_velocity_sequences",
                    "eighty_percent_phase_distance_velocity_sequences",
                    "largest_deviation_point_value_phase_distance_velocity_sequences",
                    "time_to_reach_maximum_value_phase_distance_velocity_sequences",
                    "time_to_reach_minimum_value_phase_distance_velocity_sequences",
                    "length_to_reach_maximum_value_phase_distance_velocity_sequences",
                    "length_to_reach_minimum_value_phase_distance_velocity_sequences",
                    "length_of_maximum_value_to_end_phase_distance_velocity_sequences",
                    "length_of_minimum_value_to_end_phase_distance_velocity_sequences",
                    "skewness_phase_distance_velocity_sequences",
                    "variance_phase_distance_velocity_sequences",
                    "coefcient_of_variation_phase_distance_velocity_sequences",
                    "standard_error_of_the_mean_phase_distance_velocity_sequences",
                    "interquartile_range_phase_distance_velocity_sequences",
                    "median_absolute_deviation_phase_distance_velocity_sequences",
                    "median_value_at_first_3_points_phase_distance_velocity_sequences",
                    "median_value_at_first_5_points_phase_distance_velocity_sequences",
                    "median_value_at_last_3_points_phase_distance_velocity_sequences",
                    "median_value_at_last_5_points_phase_distance_velocity_sequences",
                    "maximum_value_point_to_end_point_time_phase_distance_velocity_sequences",
                    "minimum_value_point_to_end_point_time_phase_distance_velocity_sequences",
                    "maximum_value_portion_phase_distance_velocity_sequences",
                    "minimum_value_portion_phase_distance_velocity_sequences",
                    "kurtosis_phase_distance_acc_sequences",
                    "mean_phase_distance_acc_sequences",
                    "mean_before_max_deviation_point_phase_distance_acc_sequences",
                    "mean_after_max_deviation_point_phase_distance_acc_sequences",
                    "minimum_phase_distance_acc_sequences",
                    "maximum_phase_distance_acc_sequences",
                    "standard_deviation_phase_distance_acc_sequences",
                    "median_phase_distance_acc_sequences",
                    "low_quartile_phase_distance_acc_sequences",
                    "second_quartile_phase_distance_acc_sequences",
                    "third_quartile_phase_distance_acc_sequences",
                    "quadratic_mean_phase_distance_acc_sequences",
                    "harmonic_mean_phase_distance_acc_sequences",
                    "geometric_mean_phase_distance_acc_sequences",
                    "mean_absolute_deviation_phase_distance_acc_sequences",
                    "first_point_value_phase_distance_acc_sequences",
                    "last_point_value_phase_distance_acc_sequences",
                    "twenty_percent_phase_distance_acc_sequences",
                    "eighty_percent_phase_distance_acc_sequences",
                    "largest_deviation_point_value_phase_distance_acc_sequences",
                    "time_to_reach_maximum_value_phase_distance_acc_sequences",
                    "time_to_reach_minimum_value_phase_distance_acc_sequences",
                    "length_to_reach_maximum_value_phase_distance_acc_sequences",
                    "length_to_reach_minimum_value_phase_distance_acc_sequences",
                    "length_of_maximum_value_to_end_phase_distance_acc_sequences",
                    "length_of_minimum_value_to_end_phase_distance_acc_sequences",
                    "skewness_phase_distance_acc_sequences",
                    "variance_phase_distance_acc_sequences",
                    "coefcient_of_variation_phase_distance_acc_sequences",
                    "standard_error_of_the_mean_phase_distance_acc_sequences",
                    "interquartile_range_phase_distance_acc_sequences",
                    "median_absolute_deviation_phase_distance_acc_sequences",
                    "median_value_at_first_3_points_phase_distance_acc_sequences",
                    "median_value_at_first_5_points_phase_distance_acc_sequences",
                    "median_value_at_last_3_points_phase_distance_acc_sequences",
                    "median_value_at_last_5_points_phase_distance_acc_sequences",
                    "maximum_value_point_to_end_point_time_phase_distance_acc_sequences",
                    "minimum_value_point_to_end_point_time_phase_distance_acc_sequences",
                    "maximum_value_portion_phase_distance_acc_sequences",
                    "minimum_value_portion_phase_distance_acc_sequences",
                    "kurtosis_pressure_sequences",
                    "mean_pressure_sequences",
                    "mean_before_max_deviation_point_pressure_sequences",
                    "mean_after_max_deviation_point_pressure_sequences",
                    "minimum_pressure_sequences",
                    "maximum_pressure_sequences",
                    "standard_deviation_pressure_sequences",
                    "median_pressure_sequences",
                    "low_quartile_pressure_sequences",
                    "second_quartile_pressure_sequences",
                    "third_quartile_pressure_sequences",
                    "quadratic_mean_pressure_sequences",
                    "harmonic_mean_pressure_sequences",
                    "geometric_mean_pressure_sequences",
                    "mean_absolute_deviation_pressure_sequences",
                    "first_point_value_pressure_sequences",
                    "last_point_value_pressure_sequences",
                    "twenty_percent_pressure_sequences",
                    "eighty_percent_pressure_sequences",
                    "largest_deviation_point_value_pressure_sequences",
                    "time_to_reach_maximum_value_pressure_sequences",
                    "time_to_reach_minimum_value_pressure_sequences",
                    "length_to_reach_maximum_value_pressure_sequences",
                    "length_to_reach_minimum_value_pressure_sequences",
                    "length_of_maximum_value_to_end_pressure_sequences",
                    "length_of_minimum_value_to_end_pressure_sequences",
                    "skewness_pressure_sequences",
                    "variance_pressure_sequences",
                    "coefcient_of_variation_pressure_sequences",
                    "standard_error_of_the_mean_pressure_sequences",
                    "interquartile_range_pressure_sequences",
                    "median_absolute_deviation_pressure_sequences",
                    "median_value_at_first_3_points_pressure_sequences",
                    "median_value_at_first_5_points_pressure_sequences",
                    "median_value_at_last_3_points_pressure_sequences",
                    "median_value_at_last_5_points_pressure_sequences",
                    "maximum_value_point_to_end_point_time_pressure_sequences",
                    "minimum_value_point_to_end_point_time_pressure_sequences",
                    "maximum_value_portion_pressure_sequences",
                    "minimum_value_portion_pressure_sequences",
                    "kurtosis_pressure_velocity_sequences",
                    "mean_pressure_velocity_sequences",
                    "mean_before_max_deviation_point_pressure_velocity_sequences",
                    "mean_after_max_deviation_point_pressure_velocity_sequences",
                    "minimum_pressure_velocity_sequences",
                    "maximum_pressure_velocity_sequences",
                    "standard_deviation_pressure_velocity_sequences",
                    "median_pressure_velocity_sequences",
                    "low_quartile_pressure_velocity_sequences",
                    "second_quartile_pressure_velocity_sequences",
                    "third_quartile_pressure_velocity_sequences",
                    "quadratic_mean_pressure_velocity_sequences",
                    "harmonic_mean_pressure_velocity_sequences",
                    "geometric_mean_pressure_velocity_sequences",
                    "mean_absolute_deviation_pressure_velocity_sequences",
                    "first_point_value_pressure_velocity_sequences",
                    "last_point_value_pressure_velocity_sequences",
                    "twenty_percent_pressure_velocity_sequences",
                    "eighty_percent_pressure_velocity_sequences",
                    "largest_deviation_point_value_pressure_velocity_sequences",
                    "time_to_reach_maximum_value_pressure_velocity_sequences",
                    "time_to_reach_minimum_value_pressure_velocity_sequences",
                    "length_to_reach_maximum_value_pressure_velocity_sequences",
                    "length_to_reach_minimum_value_pressure_velocity_sequences",
                    "length_of_maximum_value_to_end_pressure_velocity_sequences",
                    "length_of_minimum_value_to_end_pressure_velocity_sequences",
                    "skewness_pressure_velocity_sequences",
                    "variance_pressure_velocity_sequences",
                    "coefcient_of_variation_pressure_velocity_sequences",
                    "standard_error_of_the_mean_pressure_velocity_sequences",
                    "interquartile_range_pressure_velocity_sequences",
                    "median_absolute_deviation_pressure_velocity_sequences",
                    "median_value_at_first_3_points_pressure_velocity_sequences",
                    "median_value_at_first_5_points_pressure_velocity_sequences",
                    "median_value_at_last_3_points_pressure_velocity_sequences",
                    "median_value_at_last_5_points_pressure_velocity_sequences",
                    "maximum_value_point_to_end_point_time_pressure_velocity_sequences",
                    "minimum_value_point_to_end_point_time_pressure_velocity_sequences",
                    "maximum_value_portion_pressure_velocity_sequences",
                    "minimum_value_portion_pressure_velocity_sequences",
                    "kurtosis_pressure_acc_sequences",
                    "mean_pressure_acc_sequences",
                    "mean_before_max_deviation_point_pressure_acc_sequences",
                    "mean_after_max_deviation_point_pressure_acc_sequences",
                    "minimum_pressure_acc_sequences",
                    "maximum_pressure_acc_sequences",
                    "standard_deviation_pressure_acc_sequences",
                    "median_pressure_acc_sequences",
                    "low_quartile_pressure_acc_sequences",
                    "second_quartile_pressure_acc_sequences",
                    "third_quartile_pressure_acc_sequences",
                    "quadratic_mean_pressure_acc_sequences",
                    "harmonic_mean_pressure_acc_sequences",
                    "geometric_mean_pressure_acc_sequences",
                    "mean_absolute_deviation_pressure_acc_sequences",
                    "first_point_value_pressure_acc_sequences",
                    "last_point_value_pressure_acc_sequences",
                    "twenty_percent_pressure_acc_sequences",
                    "eighty_percent_pressure_acc_sequences",
                    "largest_deviation_point_value_pressure_acc_sequences",
                    "time_to_reach_maximum_value_pressure_acc_sequences",
                    "time_to_reach_minimum_value_pressure_acc_sequences",
                    "length_to_reach_maximum_value_pressure_acc_sequences",
                    "length_to_reach_minimum_value_pressure_acc_sequences",
                    "length_of_maximum_value_to_end_pressure_acc_sequences",
                    "length_of_minimum_value_to_end_pressure_acc_sequences",
                    "skewness_pressure_acc_sequences",
                    "variance_pressure_acc_sequences",
                    "coefcient_of_variation_pressure_acc_sequences",
                    "standard_error_of_the_mean_pressure_acc_sequences",
                    "interquartile_range_pressure_acc_sequences",
                    "median_absolute_deviation_pressure_acc_sequences",
                    "median_value_at_first_3_points_pressure_acc_sequences",
                    "median_value_at_first_5_points_pressure_acc_sequences",
                    "median_value_at_last_3_points_pressure_acc_sequences",
                    "median_value_at_last_5_points_pressure_acc_sequences",
                    "maximum_value_point_to_end_point_time_pressure_acc_sequences",
                    "minimum_value_point_to_end_point_time_pressure_acc_sequences",
                    "maximum_value_portion_pressure_acc_sequences",
                    "minimum_value_portion_pressure_acc_sequences",
                    "kurtosis_area_sequences",
                    "mean_area_sequences",
                    "mean_before_max_deviation_point_area_sequences",
                    "mean_after_max_deviation_point_area_sequences",
                    "minimum_area_sequences",
                    "maximum_area_sequences",
                    "standard_deviation_area_sequences",
                    "median_area_sequences",
                    "low_quartile_area_sequences",
                    "second_quartile_area_sequences",
                    "third_quartile_area_sequences",
                    "quadratic_mean_area_sequences",
                    "harmonic_mean_area_sequences",
                    "geometric_mean_area_sequences",
                    "mean_absolute_deviation_area_sequences",
                    "first_point_value_area_sequences",
                    "last_point_value_area_sequences",
                    "twenty_percent_area_sequences",
                    "eighty_percent_area_sequences",
                    "largest_deviation_point_value_area_sequences",
                    "time_to_reach_maximum_value_area_sequences",
                    "time_to_reach_minimum_value_area_sequences",
                    "length_to_reach_maximum_value_area_sequences",
                    "length_to_reach_minimum_value_area_sequences",
                    "length_of_maximum_value_to_end_area_sequences",
                    "length_of_minimum_value_to_end_area_sequences",
                    "skewness_area_sequences",
                    "variance_area_sequences",
                    "coefcient_of_variation_area_sequences",
                    "standard_error_of_the_mean_area_sequences",
                    "interquartile_range_area_sequences",
                    "median_absolute_deviation_area_sequences",
                    "median_value_at_first_3_points_area_sequences",
                    "median_value_at_first_5_points_area_sequences",
                    "median_value_at_last_3_points_area_sequences",
                    "median_value_at_last_5_points_area_sequences",
                    "maximum_value_point_to_end_point_time_area_sequences",
                    "minimum_value_point_to_end_point_time_area_sequences",
                    "maximum_value_portion_area_sequences",
                    "minimum_value_portion_area_sequences",
                    "kurtosis_area_velocity_sequences",
                    "mean_area_velocity_sequences",
                    "mean_before_max_deviation_point_area_velocity_sequences",
                    "mean_after_max_deviation_point_area_velocity_sequences",
                    "minimum_area_velocity_sequences",
                    "maximum_area_velocity_sequences",
                    "standard_deviation_area_velocity_sequences",
                    "median_area_velocity_sequences",
                    "low_quartile_area_velocity_sequences",
                    "second_quartile_area_velocity_sequences",
                    "third_quartile_area_velocity_sequences",
                    "quadratic_mean_area_velocity_sequences",
                    "harmonic_mean_area_velocity_sequences",
                    "geometric_mean_area_velocity_sequences",
                    "mean_absolute_deviation_area_velocity_sequences",
                    "first_point_value_area_velocity_sequences",
                    "last_point_value_area_velocity_sequences",
                    "twenty_percent_area_velocity_sequences",
                    "eighty_percent_area_velocity_sequences",
                    "largest_deviation_point_value_area_velocity_sequences",
                    "time_to_reach_maximum_value_area_velocity_sequences",
                    "time_to_reach_minimum_value_area_velocity_sequences",
                    "length_to_reach_maximum_value_area_velocity_sequences",
                    "length_to_reach_minimum_value_area_velocity_sequences",
                    "length_of_maximum_value_to_end_area_velocity_sequences",
                    "length_of_minimum_value_to_end_area_velocity_sequences",
                    "skewness_area_velocity_sequences",
                    "variance_area_velocity_sequences",
                    "coefcient_of_variation_area_velocity_sequences",
                    "standard_error_of_the_mean_area_velocity_sequences",
                    "interquartile_range_area_velocity_sequences",
                    "median_absolute_deviation_area_velocity_sequences",
                    "median_value_at_first_3_points_area_velocity_sequences",
                    "median_value_at_first_5_points_area_velocity_sequences",
                    "median_value_at_last_3_points_area_velocity_sequences",
                    "median_value_at_last_5_points_area_velocity_sequences",
                    "maximum_value_point_to_end_point_time_area_velocity_sequences",
                    "minimum_value_point_to_end_point_time_area_velocity_sequences",
                    "maximum_value_portion_area_velocity_sequences",
                    "minimum_value_portion_area_velocity_sequences",
                    "kurtosis_area_acc_sequences",
                    "mean_area_acc_sequences",
                    "mean_before_max_deviation_point_area_acc_sequences",
                    "mean_after_max_deviation_point_area_acc_sequences",
                    "minimum_area_acc_sequences",
                    "maximum_area_acc_sequences",
                    "standard_deviation_area_acc_sequences",
                    "median_area_acc_sequences",
                    "low_quartile_area_acc_sequences",
                    "second_quartile_area_acc_sequences",
                    "third_quartile_area_acc_sequences",
                    "quadratic_mean_area_acc_sequences",
                    "harmonic_mean_area_acc_sequences",
                    "geometric_mean_area_acc_sequences",
                    "mean_absolute_deviation_area_acc_sequences",
                    "first_point_value_area_acc_sequences",
                    "last_point_value_area_acc_sequences",
                    "twenty_percent_area_acc_sequences",
                    "eighty_percent_area_acc_sequences",
                    "largest_deviation_point_value_area_acc_sequences",
                    "time_to_reach_maximum_value_area_acc_sequences",
                    "time_to_reach_minimum_value_area_acc_sequences",
                    "length_to_reach_maximum_value_area_acc_sequences",
                    "length_to_reach_minimum_value_area_acc_sequences",
                    "length_of_maximum_value_to_end_area_acc_sequences",
                    "length_of_minimum_value_to_end_area_acc_sequences",
                    "skewness_area_acc_sequences",
                    "variance_area_acc_sequences",
                    "coefcient_of_variation_area_acc_sequences",
                    "standard_error_of_the_mean_area_acc_sequences",
                    "interquartile_range_area_acc_sequences",
                    "median_absolute_deviation_area_acc_sequences",
                    "median_value_at_first_3_points_area_acc_sequences",
                    "median_value_at_first_5_points_area_acc_sequences",
                    "median_value_at_last_3_points_area_acc_sequences",
                    "median_value_at_last_5_points_area_acc_sequences",
                    "maximum_value_point_to_end_point_time_area_acc_sequences",
                    "minimum_value_point_to_end_point_time_area_acc_sequences",
                    "maximum_value_portion_area_acc_sequences",
                    "minimum_value_portion_area_acc_sequences",
                    "kurtosis_angular_sequences",
                    "mean_angular_sequences",
                    "mean_before_max_deviation_point_angular_sequences",
                    "mean_after_max_deviation_point_angular_sequences",
                    "minimum_angular_sequences",
                    "maximum_angular_sequences",
                    "standard_deviation_angular_sequences",
                    "median_angular_sequences",
                    "low_quartile_angular_sequences",
                    "second_quartile_angular_sequences",
                    "third_quartile_angular_sequences",
                    "quadratic_mean_angular_sequences",
                    "harmonic_mean_angular_sequences",
                    "geometric_mean_angular_sequences",
                    "mean_absolute_deviation_angular_sequences",
                    "first_point_value_angular_sequences",
                    "last_point_value_angular_sequences",
                    "twenty_percent_angular_sequences",
                    "eighty_percent_angular_sequences",
                    "largest_deviation_point_value_angular_sequences",
                    "time_to_reach_maximum_value_angular_sequences",
                    "time_to_reach_minimum_value_angular_sequences",
                    "length_to_reach_maximum_value_angular_sequences",
                    "length_to_reach_minimum_value_angular_sequences",
                    "length_of_maximum_value_to_end_angular_sequences",
                    "length_of_minimum_value_to_end_angular_sequences",
                    "skewness_angular_sequences",
                    "variance_angular_sequences",
                    "coefcient_of_variation_angular_sequences",
                    "standard_error_of_the_mean_angular_sequences",
                    "interquartile_range_angular_sequences",
                    "median_absolute_deviation_angular_sequences",
                    "median_value_at_first_3_points_angular_sequences",
                    "median_value_at_first_5_points_angular_sequences",
                    "median_value_at_last_3_points_angular_sequences",
                    "median_value_at_last_5_points_angular_sequences",
                    "maximum_value_point_to_end_point_time_angular_sequences",
                    "minimum_value_point_to_end_point_time_angular_sequences",
                    "maximum_value_portion_angular_sequences",
                    "minimum_value_portion_angular_sequences",
                    "kurtosis_angular_velocity_sequences",
                    "mean_angular_velocity_sequences",
                    "mean_before_max_deviation_point_angular_velocity_sequences",
                    "mean_after_max_deviation_point_angular_velocity_sequences",
                    "minimum_angular_velocity_sequences",
                    "maximum_angular_velocity_sequences",
                    "standard_deviation_angular_velocity_sequences",
                    "median_angular_velocity_sequences",
                    "low_quartile_angular_velocity_sequences",
                    "second_quartile_angular_velocity_sequences",
                    "third_quartile_angular_velocity_sequences",
                    "quadratic_mean_angular_velocity_sequences",
                    "harmonic_mean_angular_velocity_sequences",
                    "geometric_mean_angular_velocity_sequences",
                    "mean_absolute_deviation_angular_velocity_sequences",
                    "first_point_value_angular_velocity_sequences",
                    "last_point_value_angular_velocity_sequences",
                    "twenty_percent_angular_velocity_sequences",
                    "eighty_percent_angular_velocity_sequences",
                    "largest_deviation_point_value_angular_velocity_sequences",
                    "time_to_reach_maximum_value_angular_velocity_sequences",
                    "time_to_reach_minimum_value_angular_velocity_sequences",
                    "length_to_reach_maximum_value_angular_velocity_sequences",
                    "length_to_reach_minimum_value_angular_velocity_sequences",
                    "length_of_maximum_value_to_end_angular_velocity_sequences",
                    "length_of_minimum_value_to_end_angular_velocity_sequences",
                    "skewness_angular_velocity_sequences",
                    "variance_angular_velocity_sequences",
                    "coefcient_of_variation_angular_velocity_sequences",
                    "standard_error_of_the_mean_angular_velocity_sequences",
                    "interquartile_range_angular_velocity_sequences",
                    "median_absolute_deviation_angular_velocity_sequences",
                    "median_value_at_first_3_points_angular_velocity_sequences",
                    "median_value_at_first_5_points_angular_velocity_sequences",
                    "median_value_at_last_3_points_angular_velocity_sequences",
                    "median_value_at_last_5_points_angular_velocity_sequences",
                    "maximum_value_point_to_end_point_time_angular_velocity_sequences",
                    "minimum_value_point_to_end_point_time_angular_velocity_sequences",
                    "maximum_value_portion_angular_velocity_sequences",
                    "minimum_value_portion_angular_velocity_sequences",
                    "kurtosis_angular_acc_sequences",
                    "mean_angular_acc_sequences",
                    "mean_before_max_deviation_point_angular_acc_sequences",
                    "mean_after_max_deviation_point_angular_acc_sequences",
                    "minimum_angular_acc_sequences",
                    "maximum_angular_acc_sequences",
                    "standard_deviation_angular_acc_sequences",
                    "median_angular_acc_sequences",
                    "low_quartile_angular_acc_sequences",
                    "second_quartile_angular_acc_sequences",
                    "third_quartile_angular_acc_sequences",
                    "quadratic_mean_angular_acc_sequences",
                    "harmonic_mean_angular_acc_sequences",
                    "geometric_mean_angular_acc_sequences",
                    "mean_absolute_deviation_angular_acc_sequences",
                    "first_point_value_angular_acc_sequences",
                    "last_point_value_angular_acc_sequences",
                    "twenty_percent_angular_acc_sequences",
                    "eighty_percent_angular_acc_sequences",
                    "largest_deviation_point_value_angular_acc_sequences",
                    "time_to_reach_maximum_value_angular_acc_sequences",
                    "time_to_reach_minimum_value_angular_acc_sequences",
                    "length_to_reach_maximum_value_angular_acc_sequences",
                    "length_to_reach_minimum_value_angular_acc_sequences",
                    "length_of_maximum_value_to_end_angular_acc_sequences",
                    "length_of_minimum_value_to_end_angular_acc_sequences",
                    "skewness_angular_acc_sequences",
                    "variance_angular_acc_sequences",
                    "coefcient_of_variation_angular_acc_sequences",
                    "standard_error_of_the_mean_angular_acc_sequences",
                    "interquartile_range_angular_acc_sequences",
                    "median_absolute_deviation_angular_acc_sequences",
                    "median_value_at_first_3_points_angular_acc_sequences",
                    "median_value_at_first_5_points_angular_acc_sequences",
                    "median_value_at_last_3_points_angular_acc_sequences",
                    "median_value_at_last_5_points_angular_acc_sequences",
                    "maximum_value_point_to_end_point_time_angular_acc_sequences",
                    "minimum_value_point_to_end_point_time_angular_acc_sequences",
                    "maximum_value_portion_angular_acc_sequences",
                    "minimum_value_portion_angular_acc_sequences",
                    "kurtosis_pairwDist_sequences",
                    "mean_pairwDist_sequences",
                    "mean_before_max_deviation_point_pairwDist_sequences",
                    "mean_after_max_deviation_point_pairwDist_sequences",
                    "minimum_pairwDist_sequences",
                    "maximum_pairwDist_sequences",
                    "standard_deviation_pairwDist_sequences",
                    "median_pairwDist_sequences",
                    "low_quartile_pairwDist_sequences",
                    "second_quartile_pairwDist_sequences",
                    "third_quartile_pairwDist_sequences",
                    "quadratic_mean_pairwDist_sequences",
                    "harmonic_mean_pairwDist_sequences",
                    "geometric_mean_pairwDist_sequences",
                    "mean_absolute_deviation_pairwDist_sequences",
                    "first_point_value_pairwDist_sequences",
                    "last_point_value_pairwDist_sequences",
                    "twenty_percent_pairwDist_sequences",
                    "eighty_percent_pairwDist_sequences",
                    "largest_deviation_point_value_pairwDist_sequences",
                    "time_to_reach_maximum_value_pairwDist_sequences",
                    "time_to_reach_minimum_value_pairwDist_sequences",
                    "length_to_reach_maximum_value_pairwDist_sequences",
                    "length_to_reach_minimum_value_pairwDist_sequences",
                    "length_of_maximum_value_to_end_pairwDist_sequences",
                    "length_of_minimum_value_to_end_pairwDist_sequences",
                    "skewness_pairwDist_sequences",
                    "variance_pairwDist_sequences",
                    "coefcient_of_variation_pairwDist_sequences",
                    "standard_error_of_the_mean_pairwDist_sequences",
                    "interquartile_range_pairwDist_sequences",
                    "median_absolute_deviation_pairwDist_sequences",
                    "median_value_at_first_3_points_pairwDist_sequences",
                    "median_value_at_first_5_points_pairwDist_sequences",
                    "median_value_at_last_3_points_pairwDist_sequences",
                    "median_value_at_last_5_points_pairwDist_sequences",
                    "maximum_value_point_to_end_point_time_pairwDist_sequences",
                    "minimum_value_point_to_end_point_time_pairwDist_sequences",
                    "maximum_value_portion_pairwDist_sequences",
                    "minimum_value_portion_pairwDist_sequences",
                    "kurtosis_xDisp_sequences",
                    "mean_xDisp_sequences",
                    "mean_before_max_deviation_point_xDisp_sequences",
                    "mean_after_max_deviation_point_xDisp_sequences",
                    "minimum_xDisp_sequences",
                    "maximum_xDisp_sequences",
                    "standard_deviation_xDisp_sequences",
                    "median_xDisp_sequences",
                    "low_quartile_xDisp_sequences",
                    "second_quartile_xDisp_sequences",
                    "third_quartile_xDisp_sequences",
                    "quadratic_mean_xDisp_sequences",
                    "harmonic_mean_xDisp_sequences",
                    "geometric_mean_xDisp_sequences",
                    "mean_absolute_deviation_xDisp_sequences",
                    "first_point_value_xDisp_sequences",
                    "last_point_value_xDisp_sequences",
                    "twenty_percent_xDisp_sequences",
                    "eighty_percent_xDisp_sequences",
                    "largest_deviation_point_value_xDisp_sequences",
                    "time_to_reach_maximum_value_xDisp_sequences",
                    "time_to_reach_minimum_value_xDisp_sequences",
                    "length_to_reach_maximum_value_xDisp_sequences",
                    "length_to_reach_minimum_value_xDisp_sequences",
                    "length_of_maximum_value_to_end_xDisp_sequences",
                    "length_of_minimum_value_to_end_xDisp_sequences",
                    "skewness_xDisp_sequences",
                    "variance_xDisp_sequences",
                    "coefcient_of_variation_xDisp_sequences",
                    "standard_error_of_the_mean_xDisp_sequences",
                    "interquartile_range_xDisp_sequences",
                    "median_absolute_deviation_xDisp_sequences",
                    "median_value_at_first_3_points_xDisp_sequences",
                    "median_value_at_first_5_points_xDisp_sequences",
                    "median_value_at_last_3_points_xDisp_sequences",
                    "median_value_at_last_5_points_xDisp_sequences",
                    "maximum_value_point_to_end_point_time_xDisp_sequences",
                    "minimum_value_point_to_end_point_time_xDisp_sequences",
                    "maximum_value_portion_xDisp_sequences",
                    "minimum_value_portion_xDisp_sequences",
                    "kurtosis_yDisp_sequences",
                    "mean_yDisp_sequences",
                    "mean_before_max_deviation_point_yDisp_sequences",
                    "mean_after_max_deviation_point_yDisp_sequences",
                    "minimum_yDisp_sequences",
                    "maximum_yDisp_sequences",
                    "standard_deviation_yDisp_sequences",
                    "median_yDisp_sequences",
                    "low_quartile_yDisp_sequences",
                    "second_quartile_yDisp_sequences",
                    "third_quartile_yDisp_sequences",
                    "quadratic_mean_yDisp_sequences",
                    "harmonic_mean_yDisp_sequences",
                    "geometric_mean_yDisp_sequences",
                    "mean_absolute_deviation_yDisp_sequences",
                    "first_point_value_yDisp_sequences",
                    "last_point_value_yDisp_sequences",
                    "twenty_percent_yDisp_sequences",
                    "eighty_percent_yDisp_sequences",
                    "largest_deviation_point_value_yDisp_sequences",
                    "time_to_reach_maximum_value_yDisp_sequences",
                    "time_to_reach_minimum_value_yDisp_sequences",
                    "length_to_reach_maximum_value_yDisp_sequences",
                    "length_to_reach_minimum_value_yDisp_sequences",
                    "length_of_maximum_value_to_end_yDisp_sequences",
                    "length_of_minimum_value_to_end_yDisp_sequences",
                    "skewness_yDisp_sequences",
                    "variance_yDisp_sequences",
                    "coefcient_of_variation_yDisp_sequences",
                    "standard_error_of_the_mean_yDisp_sequences",
                    "interquartile_range_yDisp_sequences",
                    "median_absolute_deviation_yDisp_sequences",
                    "median_value_at_first_3_points_yDisp_sequences",
                    "median_value_at_first_5_points_yDisp_sequences",
                    "median_value_at_last_3_points_yDisp_sequences",
                    "median_value_at_last_5_points_yDisp_sequences",
                    "maximum_value_point_to_end_point_time_yDisp_sequences",
                    "minimum_value_point_to_end_point_time_yDisp_sequences",
                    "maximum_value_portion_yDisp_sequences",
                    "minimum_value_portion_yDisp_sequences",
                    "kurtosis_velocity_sequences",
                    "mean_velocity_sequences",
                    "mean_before_max_deviation_point_velocity_sequences",
                    "mean_after_max_deviation_point_velocity_sequences",
                    "minimum_velocity_sequences",
                    "maximum_velocity_sequences",
                    "standard_deviation_velocity_sequences",
                    "median_velocity_sequences",
                    "low_quartile_velocity_sequences",
                    "second_quartile_velocity_sequences",
                    "third_quartile_velocity_sequences",
                    "quadratic_mean_velocity_sequences",
                    "harmonic_mean_velocity_sequences",
                    "geometric_mean_velocity_sequences",
                    "mean_absolute_deviation_velocity_sequences",
                    "first_point_value_velocity_sequences",
                    "last_point_value_velocity_sequences",
                    "twenty_percent_velocity_sequences",
                    "eighty_percent_velocity_sequences",
                    "largest_deviation_point_value_velocity_sequences",
                    "time_to_reach_maximum_value_velocity_sequences",
                    "time_to_reach_minimum_value_velocity_sequences",
                    "length_to_reach_maximum_value_velocity_sequences",
                    "length_to_reach_minimum_value_velocity_sequences",
                    "length_of_maximum_value_to_end_velocity_sequences",
                    "length_of_minimum_value_to_end_velocity_sequences",
                    "skewness_velocity_sequences",
                    "variance_velocity_sequences",
                    "coefcient_of_variation_velocity_sequences",
                    "standard_error_of_the_mean_velocity_sequences",
                    "interquartile_range_velocity_sequences",
                    "median_absolute_deviation_velocity_sequences",
                    "median_value_at_first_3_points_velocity_sequences",
                    "median_value_at_first_5_points_velocity_sequences",
                    "median_value_at_last_3_points_velocity_sequences",
                    "median_value_at_last_5_points_velocity_sequences",
                    "maximum_value_point_to_end_point_time_velocity_sequences",
                    "minimum_value_point_to_end_point_time_velocity_sequences",
                    "maximum_value_portion_velocity_sequences",
                    "minimum_value_portion_velocity_sequences",
                    "kurtosis_x_velocity_sequences",
                    "mean_x_velocity_sequences",
                    "mean_before_max_deviation_point_x_velocity_sequences",
                    "mean_after_max_deviation_point_x_velocity_sequences",
                    "minimum_x_velocity_sequences",
                    "maximum_x_velocity_sequences",
                    "standard_deviation_x_velocity_sequences",
                    "median_x_velocity_sequences",
                    "low_quartile_x_velocity_sequences",
                    "second_quartile_x_velocity_sequences",
                    "third_quartile_x_velocity_sequences",
                    "quadratic_mean_x_velocity_sequences",
                    "harmonic_mean_x_velocity_sequences",
                    "geometric_mean_x_velocity_sequences",
                    "mean_absolute_deviation_x_velocity_sequences",
                    "first_point_value_x_velocity_sequences",
                    "last_point_value_x_velocity_sequences",
                    "twenty_percent_x_velocity_sequences",
                    "eighty_percent_x_velocity_sequences",
                    "largest_deviation_point_value_x_velocity_sequences",
                    "time_to_reach_maximum_value_x_velocity_sequences",
                    "time_to_reach_minimum_value_x_velocity_sequences",
                    "length_to_reach_maximum_value_x_velocity_sequences",
                    "length_to_reach_minimum_value_x_velocity_sequences",
                    "length_of_maximum_value_to_end_x_velocity_sequences",
                    "length_of_minimum_value_to_end_x_velocity_sequences",
                    "skewness_x_velocity_sequences",
                    "variance_x_velocity_sequences",
                    "coefcient_of_variation_x_velocity_sequences",
                    "standard_error_of_the_mean_x_velocity_sequences",
                    "interquartile_range_x_velocity_sequences",
                    "median_absolute_deviation_x_velocity_sequences",
                    "median_value_at_first_3_points_x_velocity_sequences",
                    "median_value_at_first_5_points_x_velocity_sequences",
                    "median_value_at_last_3_points_x_velocity_sequences",
                    "median_value_at_last_5_points_x_velocity_sequences",
                    "maximum_value_point_to_end_point_time_x_velocity_sequences",
                    "minimum_value_point_to_end_point_time_x_velocity_sequences",
                    "maximum_value_portion_x_velocity_sequences",
                    "minimum_value_portion_x_velocity_sequences",
                    "kurtosis_y_velocity_sequences",
                    "mean_y_velocity_sequences",
                    "mean_before_max_deviation_point_y_velocity_sequences",
                    "mean_after_max_deviation_point_y_velocity_sequences",
                    "minimum_y_velocity_sequences",
                    "maximum_y_velocity_sequences",
                    "standard_deviation_y_velocity_sequences",
                    "median_y_velocity_sequences",
                    "low_quartile_y_velocity_sequences",
                    "second_quartile_y_velocity_sequences",
                    "third_quartile_y_velocity_sequences",
                    "quadratic_mean_y_velocity_sequences",
                    "harmonic_mean_y_velocity_sequences",
                    "geometric_mean_y_velocity_sequences",
                    "mean_absolute_deviation_y_velocity_sequences",
                    "first_point_value_y_velocity_sequences",
                    "last_point_value_y_velocity_sequences",
                    "twenty_percent_y_velocity_sequences",
                    "eighty_percent_y_velocity_sequences",
                    "largest_deviation_point_value_y_velocity_sequences",
                    "time_to_reach_maximum_value_y_velocity_sequences",
                    "time_to_reach_minimum_value_y_velocity_sequences",
                    "length_to_reach_maximum_value_y_velocity_sequences",
                    "length_to_reach_minimum_value_y_velocity_sequences",
                    "length_of_maximum_value_to_end_y_velocity_sequences",
                    "length_of_minimum_value_to_end_y_velocity_sequences",
                    "skewness_y_velocity_sequences",
                    "variance_y_velocity_sequences",
                    "coefcient_of_variation_y_velocity_sequences",
                    "standard_error_of_the_mean_y_velocity_sequences",
                    "interquartile_range_y_velocity_sequences",
                    "median_absolute_deviation_y_velocity_sequences",
                    "median_value_at_first_3_points_y_velocity_sequences",
                    "median_value_at_first_5_points_y_velocity_sequences",
                    "median_value_at_last_3_points_y_velocity_sequences",
                    "median_value_at_last_5_points_y_velocity_sequences",
                    "maximum_value_point_to_end_point_time_y_velocity_sequences",
                    "minimum_value_point_to_end_point_time_y_velocity_sequences",
                    "maximum_value_portion_y_velocity_sequences",
                    "minimum_value_portion_y_velocity_sequences",
                    "kurtosis_acc_sequences",
                    "mean_acc_sequences",
                    "mean_before_max_deviation_point_acc_sequences",
                    "mean_after_max_deviation_point_acc_sequences",
                    "minimum_acc_sequences",
                    "maximum_acc_sequences",
                    "standard_deviation_acc_sequences",
                    "median_acc_sequences",
                    "low_quartile_acc_sequences",
                    "second_quartile_acc_sequences",
                    "third_quartile_acc_sequences",
                    "quadratic_mean_acc_sequences",
                    "harmonic_mean_acc_sequences",
                    "geometric_mean_acc_sequences",
                    "mean_absolute_deviation_acc_sequences",
                    "first_point_value_acc_sequences",
                    "last_point_value_acc_sequences",
                    "twenty_percent_acc_sequences",
                    "eighty_percent_acc_sequences",
                    "largest_deviation_point_value_acc_sequences",
                    "time_to_reach_maximum_value_acc_sequences",
                    "time_to_reach_minimum_value_acc_sequences",
                    "length_to_reach_maximum_value_acc_sequences",
                    "length_to_reach_minimum_value_acc_sequences",
                    "length_of_maximum_value_to_end_acc_sequences",
                    "length_of_minimum_value_to_end_acc_sequences",
                    "skewness_acc_sequences",
                    "variance_acc_sequences",
                    "coefcient_of_variation_acc_sequences",
                    "standard_error_of_the_mean_acc_sequences",
                    "interquartile_range_acc_sequences",
                    "median_absolute_deviation_acc_sequences",
                    "median_value_at_first_3_points_acc_sequences",
                    "median_value_at_first_5_points_acc_sequences",
                    "median_value_at_last_3_points_acc_sequences",
                    "median_value_at_last_5_points_acc_sequences",
                    "maximum_value_point_to_end_point_time_acc_sequences",
                    "minimum_value_point_to_end_point_time_acc_sequences",
                    "maximum_value_portion_acc_sequences",
                    "minimum_value_portion_acc_sequences",
                    "kurtosis_accX_sequences",
                    "mean_accX_sequences",
                    "mean_before_max_deviation_point_accX_sequences",
                    "mean_after_max_deviation_point_accX_sequences",
                    "minimum_accX_sequences",
                    "maximum_accX_sequences",
                    "standard_deviation_accX_sequences",
                    "median_accX_sequences",
                    "low_quartile_accX_sequences",
                    "second_quartile_accX_sequences",
                    "third_quartile_accX_sequences",
                    "quadratic_mean_accX_sequences",
                    "harmonic_mean_accX_sequences",
                    "geometric_mean_accX_sequences",
                    "mean_absolute_deviation_accX_sequences",
                    "first_point_value_accX_sequences",
                    "last_point_value_accX_sequences",
                    "twenty_percent_accX_sequences",
                    "eighty_percent_accX_sequences",
                    "largest_deviation_point_value_accX_sequences",
                    "time_to_reach_maximum_value_accX_sequences",
                    "time_to_reach_minimum_value_accX_sequences",
                    "length_to_reach_maximum_value_accX_sequences",
                    "length_to_reach_minimum_value_accX_sequences",
                    "length_of_maximum_value_to_end_accX_sequences",
                    "length_of_minimum_value_to_end_accX_sequences",
                    "skewness_accX_sequences",
                    "variance_accX_sequences",
                    "coefcient_of_variation_accX_sequences",
                    "standard_error_of_the_mean_accX_sequences",
                    "interquartile_range_accX_sequences",
                    "median_absolute_deviation_accX_sequences",
                    "median_value_at_first_3_points_accX_sequences",
                    "median_value_at_first_5_points_accX_sequences",
                    "median_value_at_last_3_points_accX_sequences",
                    "median_value_at_last_5_points_accX_sequences",
                    "maximum_value_point_to_end_point_time_accX_sequences",
                    "minimum_value_point_to_end_point_time_accX_sequences",
                    "maximum_value_portion_accX_sequences",
                    "minimum_value_portion_accX_sequences",
                    "kurtosis_accY_sequences",
                    "mean_accY_sequences",
                    "mean_before_max_deviation_point_accY_sequences",
                    "mean_after_max_deviation_point_accY_sequences",
                    "minimum_accY_sequences",
                    "maximum_accY_sequences",
                    "standard_deviation_accY_sequences",
                    "median_accY_sequences",
                    "low_quartile_accY_sequences",
                    "second_quartile_accY_sequences",
                    "third_quartile_accY_sequences",
                    "quadratic_mean_accY_sequences",
                    "harmonic_mean_accY_sequences",
                    "geometric_mean_accY_sequences",
                    "mean_absolute_deviation_accY_sequences",
                    "first_point_value_accY_sequences",
                    "last_point_value_accY_sequences",
                    "twenty_percent_accY_sequences",
                    "eighty_percent_accY_sequences",
                    "largest_deviation_point_value_accY_sequences",
                    "time_to_reach_maximum_value_accY_sequences",
                    "time_to_reach_minimum_value_accY_sequences",
                    "length_to_reach_maximum_value_accY_sequences",
                    "length_to_reach_minimum_value_accY_sequences",
                    "length_of_maximum_value_to_end_accY_sequences",
                    "length_of_minimum_value_to_end_accY_sequences",
                    "skewness_accY_sequences",
                    "variance_accY_sequences",
                    "coefcient_of_variation_accY_sequences",
                    "standard_error_of_the_mean_accY_sequences",
                    "interquartile_range_accY_sequences",
                    "median_absolute_deviation_accY_sequences",
                    "median_value_at_first_3_points_accY_sequences",
                    "median_value_at_first_5_points_accY_sequences",
                    "median_value_at_last_3_points_accY_sequences",
                    "median_value_at_last_5_points_accY_sequences",
                    "maximum_value_point_to_end_point_time_accY_sequences",
                    "minimum_value_point_to_end_point_time_accY_sequences",
                    "maximum_value_portion_accY_sequences",
                    "minimum_value_portion_accY_sequences",
                    "kurtosis_deviation_sequences",
                    "mean_deviation_sequences",
                    "mean_before_max_deviation_point_deviation_sequences",
                    "mean_after_max_deviation_point_deviation_sequences",
                    "minimum_deviation_sequences",
                    "maximum_deviation_sequences",
                    "standard_deviation_deviation_sequences",
                    "median_deviation_sequences",
                    "low_quartile_deviation_sequences",
                    "second_quartile_deviation_sequences",
                    "third_quartile_deviation_sequences",
                    "quadratic_mean_deviation_sequences",
                    "harmonic_mean_deviation_sequences",
                    "geometric_mean_deviation_sequences",
                    "mean_absolute_deviation_deviation_sequences",
                    "first_point_value_deviation_sequences",
                    "last_point_value_deviation_sequences",
                    "twenty_percent_deviation_sequences",
                    "eighty_percent_deviation_sequences",
                    "largest_deviation_point_value_deviation_sequences",
                    "time_to_reach_maximum_value_deviation_sequences",
                    "time_to_reach_minimum_value_deviation_sequences",
                    "length_to_reach_maximum_value_deviation_sequences",
                    "length_to_reach_minimum_value_deviation_sequences",
                    "length_of_maximum_value_to_end_deviation_sequences",
                    "length_of_minimum_value_to_end_deviation_sequences",
                    "skewness_deviation_sequences",
                    "variance_deviation_sequences",
                    "coefcient_of_variation_deviation_sequences",
                    "standard_error_of_the_mean_deviation_sequences",
                    "interquartile_range_deviation_sequences",
                    "median_absolute_deviation_deviation_sequences",
                    "median_value_at_first_3_points_deviation_sequences",
                    "median_value_at_first_5_points_deviation_sequences",
                    "median_value_at_last_3_points_deviation_sequences",
                    "median_value_at_last_5_points_deviation_sequences",
                    "maximum_value_point_to_end_point_time_deviation_sequences",
                    "minimum_value_point_to_end_point_time_deviation_sequences",
                    "maximum_value_portion_deviation_sequences",
                    "minimum_value_portion_deviation_sequences",
                    "kurtosis_deviation_velocity_sequences",
                    "mean_deviation_velocity_sequences",
                    "mean_before_max_deviation_point_deviation_velocity_sequences",
                    "mean_after_max_deviation_point_deviation_velocity_sequences",
                    "minimum_deviation_velocity_sequences",
                    "maximum_deviation_velocity_sequences",
                    "standard_deviation_deviation_velocity_sequences",
                    "median_deviation_velocity_sequences",
                    "low_quartile_deviation_velocity_sequences",
                    "second_quartile_deviation_velocity_sequences",
                    "third_quartile_deviation_velocity_sequences",
                    "quadratic_mean_deviation_velocity_sequences",
                    "harmonic_mean_deviation_velocity_sequences",
                    "geometric_mean_deviation_velocity_sequences",
                    "mean_absolute_deviation_deviation_velocity_sequences",
                    "first_point_value_deviation_velocity_sequences",
                    "last_point_value_deviation_velocity_sequences",
                    "twenty_percent_deviation_velocity_sequences",
                    "eighty_percent_deviation_velocity_sequences",
                    "largest_deviation_point_value_deviation_velocity_sequences",
                    "time_to_reach_maximum_value_deviation_velocity_sequences",
                    "time_to_reach_minimum_value_deviation_velocity_sequences",
                    "length_to_reach_maximum_value_deviation_velocity_sequences",
                    "length_to_reach_minimum_value_deviation_velocity_sequences",
                    "length_of_maximum_value_to_end_deviation_velocity_sequences",
                    "length_of_minimum_value_to_end_deviation_velocity_sequences",
                    "skewness_deviation_velocity_sequences",
                    "variance_deviation_velocity_sequences",
                    "coefcient_of_variation_deviation_velocity_sequences",
                    "standard_error_of_the_mean_deviation_velocity_sequences",
                    "interquartile_range_deviation_velocity_sequences",
                    "median_absolute_deviation_deviation_velocity_sequences",
                    "median_value_at_first_3_points_deviation_velocity_sequences",
                    "median_value_at_first_5_points_deviation_velocity_sequences",
                    "median_value_at_last_3_points_deviation_velocity_sequences",
                    "median_value_at_last_5_points_deviation_velocity_sequences",
                    "maximum_value_point_to_end_point_time_deviation_velocity_sequences",
                    "minimum_value_point_to_end_point_time_deviation_velocity_sequences",
                    "maximum_value_portion_deviation_velocity_sequences",
                    "minimum_value_portion_deviation_velocity_sequences",
                    "kurtosis_deviation_acc_sequences",
                    "mean_deviation_acc_sequences",
                    "mean_before_max_deviation_point_deviation_acc_sequences",
                    "mean_after_max_deviation_point_deviation_acc_sequences",
                    "minimum_deviation_acc_sequences",
                    "maximum_deviation_acc_sequences",
                    "standard_deviation_deviation_acc_sequences",
                    "median_deviation_acc_sequences",
                    "low_quartile_deviation_acc_sequences",
                    "second_quartile_deviation_acc_sequences",
                    "third_quartile_deviation_acc_sequences",
                    "quadratic_mean_deviation_acc_sequences",
                    "harmonic_mean_deviation_acc_sequences",
                    "geometric_mean_deviation_acc_sequences",
                    "mean_absolute_deviation_deviation_acc_sequences",
                    "first_point_value_deviation_acc_sequences",
                    "last_point_value_deviation_acc_sequences",
                    "twenty_percent_deviation_acc_sequences",
                    "eighty_percent_deviation_acc_sequences",
                    "largest_deviation_point_value_deviation_acc_sequences",
                    "time_to_reach_maximum_value_deviation_acc_sequences",
                    "time_to_reach_minimum_value_deviation_acc_sequences",
                    "length_to_reach_maximum_value_deviation_acc_sequences",
                    "length_to_reach_minimum_value_deviation_acc_sequences",
                    "length_of_maximum_value_to_end_deviation_acc_sequences",
                    "length_of_minimum_value_to_end_deviation_acc_sequences",
                    "skewness_deviation_acc_sequences",
                    "variance_deviation_acc_sequences",
                    "coefcient_of_variation_deviation_acc_sequences",
                    "standard_error_of_the_mean_deviation_acc_sequences",
                    "interquartile_range_deviation_acc_sequences",
                    "median_absolute_deviation_deviation_acc_sequences",
                    "median_value_at_first_3_points_deviation_acc_sequences",
                    "median_value_at_first_5_points_deviation_acc_sequences",
                    "median_value_at_last_3_points_deviation_acc_sequences",
                    "median_value_at_last_5_points_deviation_acc_sequences",
                    "maximum_value_point_to_end_point_time_deviation_acc_sequences",
                    "minimum_value_point_to_end_point_time_deviation_acc_sequences",
                    "maximum_value_portion_deviation_acc_sequences",
                    "minimum_value_portion_deviation_acc_sequences",
                    "direction",
                ]
            ]
            getcontext().prec = 32
            #touch_data存储所有触摸数据
            touch_data = []
            #with 语句：这是Python中的上下文管理器，它用于封装一个代码块的执行，以确保在代码块执行完毕后，所占用的资源（如文件）能够被正确地关闭或释放。
            #open(path, encoding="utf-8")：打开位于path路径的文件。encoding="utf-8"参数指定文件采用UTF-8编码，这对于正确读取包含特殊字符的文件是必要的。
            #as touches：将打开的文件对象赋予变量名touches，用于在后续代码中进行操作。
            with open(path, encoding="utf-8") as touches:
                #创建CSV阅读器：csv.reader(touches, delimiter=",") 创建一个CSV阅读器对象，用于逐行读取文件内容。这里指定分隔符为逗号。
                touchreader = csv.reader(touches, delimiter=",")
                #读取标题行：order = next(touchreader) 读取CSV文件的第一行
                order = next(touchreader)
                cnv = {}
                #创建列名到索引的映射：接下来的循环遍历order列表（列标题），创建一个字典cnv，将每个列标题映射到其索引。这使得后续代码可以通过列名来访问每行数据中的相应值。
                for i, o in enumerate(order):
                    cnv[o] = i
                single_touch = []
                #记录触摸数据
                for touch in touchreader:
                    if touch[cnv["type"]] == "2":
                        single_touch.append(
                            {
                                "pressure": touch[cnv["pressure"]],
                                "timestamp": touch[cnv["timestamp"]],
                                "y": touch[cnv["y"]],
                                "x": touch[cnv["x"]],
                                "area": touch[cnv["area"]],
                            }
                        )
                        touch_data.append(single_touch)
                        single_touch = []
                    else:
                        single_touch.append(
                            {
                                "pressure": touch[cnv["pressure"]],
                                "timestamp": touch[cnv["timestamp"]],
                                "y": touch[cnv["y"]],
                                "x": touch[cnv["x"]],
                                "area": touch[cnv["area"]],
                            }
                        )
            size = width, height = 0, 0
            multiplierX = 1
            multiplierY = 1
            #屏幕大小适配比例计算
            if model in ["iPhone 6s Plus", "iPhone 7 Plus", "iPhone 8 Plus"]:
                multiplierX = 1
                multiplierY = 1
                size = width, height = 1080, 1920
            elif model in ["iPhone 6s", "iPhone 7", "iPhone 8"]:
                size = width, height = 750, 1334
                multiplierX = 1.44
                multiplierY = 1.44
            elif model in ["iPhone X", "iPhone XS"]:
                size = width, height = 1125, 2436
                multiplierX = 0.96
                multiplyerY = 0.7882
            elif model in ["iPhone XS Max"]:
                size = width, height = 1242, 2688
                multiplierX = 0.8696
                multiplyerY = 0.7143
            else:
                print(model)
            #技术器
            swipe_counter = 0
            #丢弃数据计数器
            discard_swipe_counter = 0
            #遍历每一个触摸数据，touch里面包含了一个触摸数据中的触摸点的记录
            for touch in touch_data:
                #swipeCounts：用于存储某些计数或汇总统计的列表。
                swipeCounts = []
                #percentails：用于计算速度和加速度的百分位数，这里选择了25%、50%、75%。
                percentails = [25, 50, 75]
                #emp：当前触摸事件序列。
                temp = touch
                #swipe_count：当前触摸事件序列的长度。
                swipe_count = len(temp)
                midX = []
                midY = []
                #遍历触摸实践中的每一次swipe或scroll
                for t in temp:
                    t["x"] = str(multiplierX* float(t["x"]))
                    t["y"] = str(multiplierY * float(t["y"]))
                for t in temp:
                    midX.append(float(t["x"]))
                    midY.append(float(t["y"]))
                if (statistics.stdev(midX) < 5 and statistics.stdev(midY) < 5) or len(touch) < 9:
                    #触摸轨迹中触摸点的数量小于等于6,丢弃
                    discard_swipe_counter += 1
                    continue
                else:
                    # 触摸点列表；索引最大值为t
                    pointsSequence = []
                    for t in temp:
                        pointsSequence.append([float(t["x"]), float(t["y"])])
                    #根据点集计算曲率
                    #时间列表
                    timeSequence = []
                    for t in temp:
                        timeSequence.append(float(t["timestamp"]))
                    curvatureFeatures = calculate_curvature_features(pointsSequence)
                    #持续时间
                    duration = int(
                        float(temp[-1]["timestamp"]) - float(temp[0]["timestamp"])
                    )
                    start_time = float(temp[0]["timestamp"])
                    end_time = float(temp[-1]["timestamp"])
                    tDelta = []
                    # 计算x，y位移列表和时间差;tDelta索引范围0至t-1
                    tPrev = temp[0]
                    for i, t in enumerate(temp):
                        if i != 0:
                            tDelta.append(float(t["timestamp"]) - float(tPrev["timestamp"]))
                    #开始和接受点坐标
                    startX = float(temp[0]["x"])
                    startY = float(temp[0]["y"])
                    stopX = float(temp[-1]["x"])
                    stopY = float(temp[-1]["y"])
                    displacement = math.sqrt((stopY - startY) ** 2 + (stopX - startX) ** 2)
                    tmpangle = math.atan2(stopY - startY, stopX - startX)
                    dirEndToEnd = tmpangle + math.pi
                    # dirFlag = 0
                    if tmpangle <= math.pi / 4:
                        dirFlag = 4
                    elif tmpangle > math.pi / 4 and tmpangle <= 5 * math.pi / 4:
                        if tmpangle < 3 * math.pi / 4:
                            dirFlag = 1
                        else:
                            dirFlag = 2
                    else:
                        if tmpangle < 7 * math.pi / 4:
                            dirFlag = 3
                        else:
                            dirFlag = 4
                    if direction == "swipe":
                        if startX - stopX > 0:
                            dir = "left"
                        else:
                            dir = "right"
                    elif direction == "scroll":
                        if startY - stopY > 0:
                            dir = "up"
                        else:
                            dir = "down"
                    #25%、50%、75%位置的点的坐标
                    temp25X = float(temp[len(temp) // 4]["x"])
                    temp25Y = float(temp[len(temp) // 4]["y"])
                    temp50X = float(temp[len(temp) // 2]["x"])
                    temp50Y = float(temp[len(temp) // 2]["y"])
                    temp75X = float(temp[3 * len(temp) // 4]["x"])
                    temp75Y = float(temp[3 * len(temp) // 4]["y"])
                    # 18个
                    start_to_25_point_direction = math.atan2(temp25Y - startY, temp25X - startX)
                    start_to_50_point_direction = math.atan2(temp50Y - startY, temp50X - startX)
                    start_to_75_point_direction = math.atan2(temp75Y - startY, temp75X - startX)
                    the_25_point_to_end_direction = math.atan2(stopY - temp25Y, stopX - temp25X)
                    the_50_point_to_end_direction = math.atan2(stopY - temp50Y, stopX - temp50X)
                    the_75_point_to_end_direction = math.atan2(stopY - temp75Y, stopX - temp75X)
                    #到达25%、50%、75%位置的点时所花费的时间
                    time_to_reach_25_point = float(temp[len(temp) // 4]["timestamp"]) - float(temp[0]["timestamp"])
                    time_to_reach_50_point = float(temp[len(temp) // 2]["timestamp"]) - float(temp[0]["timestamp"])
                    time_to_reach_75_point = float(temp[3 * len(temp) // 4]["timestamp"]) - float(temp[0]["timestamp"])
                    the_25_point_to_end_time = float(temp[-1]["timestamp"]) - float(temp[len(temp) // 4]["timestamp"])
                    the_50_point_to_end_time = float(temp[-1]["timestamp"]) - float(temp[len(temp) // 2]["timestamp"])
                    the_75_point_to_end_time = float(temp[-1]["timestamp"]) - float(temp[3 * len(temp) // 4]["timestamp"])
                    #起始点
                    #起始点到达25%、50%、75%位置的点时所花费的位移距离
                    length_to_reach_25_point = math.sqrt((temp25X - float(temp[0]["x"])) ** 2 + (temp25Y - float(temp[0]["y"])) ** 2)
                    length_to_reach_50_point = math.sqrt((temp50X - float(temp[0]["x"])) ** 2 + (temp50Y - float(temp[0]["y"])) ** 2)
                    length_to_reach_75_point = math.sqrt((temp75X - float(temp[0]["x"])) ** 2 + (temp75Y - float(temp[0]["y"])) ** 2)
                    the_25_point_to_end_length = math.sqrt((temp25X - float(temp[-1]["x"])) ** 2 + (temp25Y - float(temp[-1]["y"])) ** 2)
                    the_50_point_to_end_length = math.sqrt((temp50X - float(temp[-1]["x"])) ** 2 + (temp50Y - float(temp[-1]["y"])) ** 2)
                    the_75_point_to_end_length = math.sqrt((temp75X - float(temp[-1]["x"])) ** 2 + (temp75Y - float(temp[-1]["y"])) ** 2)
                    # 记录每个点到起始点至终点线段的偏离；索引最大值为t
                    projectOnPrepStraight = []
                    # 再次遍历 xVek 和 yVek，计算每个点在归一化后的垂直向量 perVek 上的投影长度，并将结果存储在 projectOnPrepStraight 列表中
                    # projectOnPrepStraight最后一个点为0
                    for i, t in enumerate(pointsSequence):
                        if i > 0 and i < (len(pointsSequence) - 1):
                            projectOnPrepStraight.append(perpendicular_distance(pointsSequence[i], pointsSequence[0],pointsSequence[len(pointsSequence) - 1]))
                    projectOnPrepStraight.insert(0, 0)
                    projectOnPrepStraight.append(0)
                    # 计算偏离速度序列,索引范围为0-t-1
                    projectOnPrepStraightVelocity = calculate_velocity(projectOnPrepStraight,tDelta, 0)
                    #计算偏离加速度序列
                    projectOnPrepStraightAcc = calculate_acc(projectOnPrepStraightVelocity, tDelta, 1)
                    maxAbs = projectOnPrepStraight[0]
                    # maxAbs记录最大偏差；abs计算绝对值
                    for proj in projectOnPrepStraight:
                        if abs(proj) > abs(maxAbs):
                            maxAbs = proj
                    # maxAbs_index代表最大偏离点的索引值；maxAbs_index范围为1至(t-1)
                    maxAbs_index = projectOnPrepStraight.index(maxAbs)
                    # 计算偏离距离序列的统计值
                    statisticsFromDeviationSequences = calculate_statistics(projectOnPrepStraight, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromDeviationVelocitySequences = calculate_statistics(projectOnPrepStraightVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromDeviationAccSequences = calculate_statistics(projectOnPrepStraightAcc, timeSequence, maxAbs_index, midX, midY)
                    start_to_largest_deviation_point_direction = math.atan2(float(temp[maxAbs_index]["y"]) - startY, float(temp[maxAbs_index]["x"])- startX)
                    largest_deviation_point_to_end_direction = math.atan2(stopY - float(temp[maxAbs_index]["y"]), stopX - float(temp[maxAbs_index]["x"]))
                    # 定义x，y位移列表；xDisp、yDisp索引范围0至t-1
                    xDisp = []
                    yDisp = []
                    # 计算x，y位移列表
                    tPrev = temp[0]
                    for i, t in enumerate(temp):
                        if i != 0:
                            xDisp.append(float(t["x"]) - float(tPrev["x"]))
                            yDisp.append(float(t["y"]) - float(tPrev["y"]))

                            tPrev = t
                    #位移列表；pairwDist索引范围为0至t-1
                    pairwDist = []
                    for i in range(len(xDisp)):
                        pairwDist.append(math.sqrt(xDisp[i] ** 2 + yDisp[i] ** 2))
                    # 计算路径长度
                    length_of_trajectory = sum(pairwDist)
                    ratio = displacement / length_of_trajectory
                    velocity_of_trajectory = length_of_trajectory / duration
                    velocity_of_displacement = displacement / duration
                    # 计算速度,时间的索引范围为0-t-1
                    # 若data的索引范围为0-t，计算得速度列表的索引范围为0-t-1,则flag = 0
                    # 若data的索引范围为0-t-1,计算得速度列表的索引范围为0-t-2,则flag = 1
                    # 若data的索引范围为0-t-1,计算得速度列表的索引范围为0-t-1,则flag = 2
                    # 若data的索引范围为0-t-2,计算得速度列表的索引范围为0-t-3,则flag = 3
                    # 计算加速度,时间的索引范围为0-t-1
                    # 若data的索引范围为0-t-2,计算得速度列表的索引范围为0-t-3,则flag = 0
                    # 若data的索引范围为0-t-1，计算得加速度列表的索引范围为0-t-2,则flag = 1
                    # 若data的索引范围为0-t-3，计算得加速度列表的索引范围为0-t-4,则flag = 2
                    pairwDistVelocity = calculate_velocity(pairwDist, tDelta, 2)
                    pairwDistVelocityX = calculate_velocity(xDisp, tDelta, 2)
                    pairwDistVelocityY = calculate_velocity(yDisp, tDelta, 2)
                    pairwDistAcc = calculate_acc(pairwDistVelocity, tDelta, 1)
                    pairwDistAccX = calculate_acc(pairwDistVelocityX, tDelta, 1)
                    pairwDistAccY = calculate_acc(pairwDistVelocityY, tDelta, 1)
                    statisticsFromPairwDist = calculate_statistics(pairwDist, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistX = calculate_statistics(xDisp, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistY = calculate_statistics(yDisp, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistVelocity = calculate_statistics(pairwDistVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistVelocityX = calculate_statistics(pairwDistVelocityX, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistVelocityY = calculate_statistics(pairwDistVelocityY, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistAcc = calculate_statistics(pairwDistAcc, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistAccX = calculate_statistics(pairwDistAccX, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPairwDistAccY = calculate_statistics(pairwDistAccY, timeSequence, maxAbs_index, midX, midY)

                    #angl索引范围为0至t-1；
                    angl = []
                    for i in range(len(xDisp)):
                        angl.append(math.atan2(yDisp[i], xDisp[i]))
                    #角度速度列表；anglVeDisp的索引范围0至t-2
                    anglVelocity = calculate_velocity(angl, tDelta, 1)
                    #角速度加速度列表
                    anglAcc = calculate_acc(anglVelocity, tDelta, 0)
                    # 计算角度变化的速度
                    statisticsFromAngl = calculate_statistics(angl, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromAnglVelocity = calculate_statistics(anglVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromAnglAcc = calculate_statistics(anglAcc, timeSequence, maxAbs_index, midX, midY)

                    #phase角度列表;pangl索引范围为0-t
                    pangl = []
                    for t in temp:
                        pangl.append(math.atan2(float(t["y"]), float(t["x"])))
                    #pangl的索引范围为0-t,pangl_v的索引范围为0-t-1
                    panglVelocity = calculate_velocity(pangl, tDelta, 0)
                    panglAcc = calculate_acc(panglVelocity, tDelta, 1)
                    # 从pangl中获取统计数据
                    statisticsFromPangl = calculate_statistics(pangl, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPanglVelocity = calculate_statistics(panglVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPanglAcc = calculate_statistics(panglAcc, timeSequence, maxAbs_index, midX, midY)

                    # 点到原点的距离
                    pointsDistance = []
                    for t in pointsSequence:
                        pointsDistance.append(math.sqrt(float(t[0]) ** 2 + float(t[1]) ** 2))
                    pointsDistanceVelocity = calculate_velocity(pointsDistance, tDelta, 0)
                    pointsDistanceAcc = calculate_acc(pointsDistanceVelocity, tDelta, 1)
                    # 统计数据：原点到触摸点位移
                    statisticsFromPointsDistance = calculate_statistics(pointsDistance, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPointsDistanceVelocity = calculate_statistics(pointsDistanceVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPointsDistanceAcc = calculate_statistics(pointsDistanceAcc, timeSequence, maxAbs_index, midX, midY)

                    pressure = []
                    for t in temp:
                        pressure.append(float(t["pressure"]))
                    pressureVelocity = calculate_velocity(pressure, tDelta, 0)
                    pressureAcc = calculate_acc(pressureVelocity, tDelta, 1)
                    statisticsFromPressure = calculate_statistics(pressure, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPressureVelocity = calculate_statistics(pressureVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromPressureAcc = calculate_statistics(pressureAcc, timeSequence, maxAbs_index, midX, midY)


                    # 面积列表
                    area = []
                    for t in temp:
                        area.append(float(t["area"]))
                    areaVelocity = calculate_velocity(area, tDelta, 0)
                    areaAcc = calculate_acc(areaVelocity, tDelta, 1)
                    # 面积速度数据序列
                    statisticsFromArea = calculate_statistics(area, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromAreaVelocity = calculate_statistics(areaVelocity, timeSequence, maxAbs_index, midX, midY)
                    statisticsFromAreaAcc = calculate_statistics(areaAcc, timeSequence, maxAbs_index, midX, midY)

                    swipeFeatures = [
                        #共个特征
                        dirFlag,
                        dirEndToEnd,
                        startX,
                        startY,
                        stopX,
                        stopY,
                        duration,
                        start_time,
                        end_time,
                        length_of_trajectory,
                        displacement,
                        ratio,
                        velocity_of_trajectory,
                        velocity_of_displacement,
                        start_to_25_point_direction,
                        start_to_50_point_direction,
                        start_to_75_point_direction,
                        the_25_point_to_end_direction,
                        the_50_point_to_end_direction,
                        the_75_point_to_end_direction,
                        start_to_largest_deviation_point_direction,
                        largest_deviation_point_to_end_direction,
                        time_to_reach_25_point,
                        time_to_reach_50_point,
                        time_to_reach_75_point,
                        the_25_point_to_end_time,
                        the_50_point_to_end_time,
                        the_75_point_to_end_time,
                        length_to_reach_25_point,
                        length_to_reach_50_point,
                        length_to_reach_75_point,
                        the_25_point_to_end_length,
                        the_50_point_to_end_length,
                        the_75_point_to_end_length,
                        curvatureFeatures[0],
                        curvatureFeatures[1],
                        statisticsFromPangl["Kurtosis"],
                        statisticsFromPangl["Mean"],
                        statisticsFromPangl["Mean Before Max Deviation Point"],
                        statisticsFromPangl["Mean After Max Deviation Point"],
                        statisticsFromPangl["Minimum"],
                        statisticsFromPangl["Maximum"],
                        statisticsFromPangl["Standard Deviation"],
                        statisticsFromPangl["Median"],
                        statisticsFromPangl["Lower Quartile"],
                        statisticsFromPangl["Second Quartile"],
                        statisticsFromPangl["Third Quartile"],
                        statisticsFromPangl["Quadratic Mean"],
                        statisticsFromPangl["Harmonic Mean"],
                        statisticsFromPangl["Geometric Mean"],
                        statisticsFromPangl["Mean Absolute Deviation"],
                        statisticsFromPangl["First Point value"],
                        statisticsFromPangl["Last Point value"],
                        statisticsFromPangl["Twenty Percent"],
                        statisticsFromPangl["Eighty Percent"],
                        statisticsFromPangl["Largest Deviation Point Value"],
                        statisticsFromPangl["Time to reach Maximum Value"],
                        statisticsFromPangl["Time to reach Minimum Value"],
                        statisticsFromPangl["Length to reach Maximum Value"],
                        statisticsFromPangl["Length to reach Minimum Value"],
                        statisticsFromPangl["Length of Maximum Value to End"],
                        statisticsFromPangl["Length of Minimum Value to End"],
                        statisticsFromPangl["Skewness"],
                        statisticsFromPangl["Variance"],
                        statisticsFromPangl["Coefcient of Variation"],
                        statisticsFromPangl["Standard Error of the Mean"],
                        statisticsFromPangl["Interquartile Range"],
                        statisticsFromPangl["Median Absolute Deviation"],
                        statisticsFromPangl["Median Value at First 3 Points"],
                        statisticsFromPangl["Median Value at First 5 Points"],
                        statisticsFromPangl["Median Value at Last 3 Points"],
                        statisticsFromPangl["Median Value at Last 5 Points"],
                        statisticsFromPangl["Maximum Value Point to End Point Time"],
                        statisticsFromPangl["Minimum Value Point to End Point Time"],
                        statisticsFromPangl["maximum_value_portion"],
                        statisticsFromPangl["minimum_value_portion"],
                        statisticsFromPanglVelocity["Kurtosis"],
                        statisticsFromPanglVelocity["Mean"],
                        statisticsFromPanglVelocity["Mean Before Max Deviation Point"],
                        statisticsFromPanglVelocity["Mean After Max Deviation Point"],
                        statisticsFromPanglVelocity["Minimum"],
                        statisticsFromPanglVelocity["Maximum"],
                        statisticsFromPanglVelocity["Standard Deviation"],
                        statisticsFromPanglVelocity["Median"],
                        statisticsFromPanglVelocity["Lower Quartile"],
                        statisticsFromPanglVelocity["Second Quartile"],
                        statisticsFromPanglVelocity["Third Quartile"],
                        statisticsFromPanglVelocity["Quadratic Mean"],
                        statisticsFromPanglVelocity["Harmonic Mean"],
                        statisticsFromPanglVelocity["Geometric Mean"],
                        statisticsFromPanglVelocity["Mean Absolute Deviation"],
                        statisticsFromPanglVelocity["First Point value"],
                        statisticsFromPanglVelocity["Last Point value"],
                        statisticsFromPanglVelocity["Twenty Percent"],
                        statisticsFromPanglVelocity["Eighty Percent"],
                        statisticsFromPanglVelocity["Largest Deviation Point Value"],
                        statisticsFromPanglVelocity["Time to reach Maximum Value"],
                        statisticsFromPanglVelocity["Time to reach Minimum Value"],
                        statisticsFromPanglVelocity["Length to reach Maximum Value"],
                        statisticsFromPanglVelocity["Length to reach Minimum Value"],
                        statisticsFromPanglVelocity["Length of Maximum Value to End"],
                        statisticsFromPanglVelocity["Length of Minimum Value to End"],
                        statisticsFromPanglVelocity["Skewness"],
                        statisticsFromPanglVelocity["Variance"],
                        statisticsFromPanglVelocity["Coefcient of Variation"],
                        statisticsFromPanglVelocity["Standard Error of the Mean"],
                        statisticsFromPanglVelocity["Interquartile Range"],
                        statisticsFromPanglVelocity["Median Absolute Deviation"],
                        statisticsFromPanglVelocity["Median Value at First 3 Points"],
                        statisticsFromPanglVelocity["Median Value at First 5 Points"],
                        statisticsFromPanglVelocity["Median Value at Last 3 Points"],
                        statisticsFromPanglVelocity["Median Value at Last 5 Points"],
                        statisticsFromPanglVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromPanglVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromPanglVelocity["maximum_value_portion"],
                        statisticsFromPanglVelocity["minimum_value_portion"],
                        statisticsFromPanglAcc["Kurtosis"],
                        statisticsFromPanglAcc["Mean"],
                        statisticsFromPanglAcc["Mean Before Max Deviation Point"],
                        statisticsFromPanglAcc["Mean After Max Deviation Point"],
                        statisticsFromPanglAcc["Minimum"],
                        statisticsFromPanglAcc["Maximum"],
                        statisticsFromPanglAcc["Standard Deviation"],
                        statisticsFromPanglAcc["Median"],
                        statisticsFromPanglAcc["Lower Quartile"],
                        statisticsFromPanglAcc["Second Quartile"],
                        statisticsFromPanglAcc["Third Quartile"],
                        statisticsFromPanglAcc["Quadratic Mean"],
                        statisticsFromPanglAcc["Harmonic Mean"],
                        statisticsFromPanglAcc["Geometric Mean"],
                        statisticsFromPanglAcc["Mean Absolute Deviation"],
                        statisticsFromPanglAcc["First Point value"],
                        statisticsFromPanglAcc["Last Point value"],
                        statisticsFromPanglAcc["Twenty Percent"],
                        statisticsFromPanglAcc["Eighty Percent"],
                        statisticsFromPanglAcc["Largest Deviation Point Value"],
                        statisticsFromPanglAcc["Time to reach Maximum Value"],
                        statisticsFromPanglAcc["Time to reach Minimum Value"],
                        statisticsFromPanglAcc["Length to reach Maximum Value"],
                        statisticsFromPanglAcc["Length to reach Minimum Value"],
                        statisticsFromPanglAcc["Length of Maximum Value to End"],
                        statisticsFromPanglAcc["Length of Minimum Value to End"],
                        statisticsFromPanglAcc["Skewness"],
                        statisticsFromPanglAcc["Variance"],
                        statisticsFromPanglAcc["Coefcient of Variation"],
                        statisticsFromPanglAcc["Standard Error of the Mean"],
                        statisticsFromPanglAcc["Interquartile Range"],
                        statisticsFromPanglAcc["Median Absolute Deviation"],
                        statisticsFromPanglAcc["Median Value at First 3 Points"],
                        statisticsFromPanglAcc["Median Value at First 5 Points"],
                        statisticsFromPanglAcc["Median Value at Last 3 Points"],
                        statisticsFromPanglAcc["Median Value at Last 5 Points"],
                        statisticsFromPanglAcc["Maximum Value Point to End Point Time"],
                        statisticsFromPanglAcc["Minimum Value Point to End Point Time"],
                        statisticsFromPanglAcc["maximum_value_portion"],
                        statisticsFromPanglAcc["minimum_value_portion"],
                        statisticsFromPointsDistance["Kurtosis"],
                        statisticsFromPointsDistance["Mean"],
                        statisticsFromPointsDistance["Mean Before Max Deviation Point"],
                        statisticsFromPointsDistance["Mean After Max Deviation Point"],
                        statisticsFromPointsDistance["Minimum"],
                        statisticsFromPointsDistance["Maximum"],
                        statisticsFromPointsDistance["Standard Deviation"],
                        statisticsFromPointsDistance["Median"],
                        statisticsFromPointsDistance["Lower Quartile"],
                        statisticsFromPointsDistance["Second Quartile"],
                        statisticsFromPointsDistance["Third Quartile"],
                        statisticsFromPointsDistance["Quadratic Mean"],
                        statisticsFromPointsDistance["Harmonic Mean"],
                        statisticsFromPointsDistance["Geometric Mean"],
                        statisticsFromPointsDistance["Mean Absolute Deviation"],
                        statisticsFromPointsDistance["First Point value"],
                        statisticsFromPointsDistance["Last Point value"],
                        statisticsFromPointsDistance["Twenty Percent"],
                        statisticsFromPointsDistance["Eighty Percent"],
                        statisticsFromPointsDistance["Largest Deviation Point Value"],
                        statisticsFromPointsDistance["Time to reach Maximum Value"],
                        statisticsFromPointsDistance["Time to reach Minimum Value"],
                        statisticsFromPointsDistance["Length to reach Maximum Value"],
                        statisticsFromPointsDistance["Length to reach Minimum Value"],
                        statisticsFromPointsDistance["Length of Maximum Value to End"],
                        statisticsFromPointsDistance["Length of Minimum Value to End"],
                        statisticsFromPointsDistance["Skewness"],
                        statisticsFromPointsDistance["Variance"],
                        statisticsFromPointsDistance["Coefcient of Variation"],
                        statisticsFromPointsDistance["Standard Error of the Mean"],
                        statisticsFromPointsDistance["Interquartile Range"],
                        statisticsFromPointsDistance["Median Absolute Deviation"],
                        statisticsFromPointsDistance["Median Value at First 3 Points"],
                        statisticsFromPointsDistance["Median Value at First 5 Points"],
                        statisticsFromPointsDistance["Median Value at Last 3 Points"],
                        statisticsFromPointsDistance["Median Value at Last 5 Points"],
                        statisticsFromPointsDistance["Maximum Value Point to End Point Time"],
                        statisticsFromPointsDistance["Minimum Value Point to End Point Time"],
                        statisticsFromPointsDistance["maximum_value_portion"],
                        statisticsFromPointsDistance["minimum_value_portion"],
                        statisticsFromPointsDistanceVelocity["Kurtosis"],
                        statisticsFromPointsDistanceVelocity["Mean"],
                        statisticsFromPointsDistanceVelocity["Mean Before Max Deviation Point"],
                        statisticsFromPointsDistanceVelocity["Mean After Max Deviation Point"],
                        statisticsFromPointsDistanceVelocity["Minimum"],
                        statisticsFromPointsDistanceVelocity["Maximum"],
                        statisticsFromPointsDistanceVelocity["Standard Deviation"],
                        statisticsFromPointsDistanceVelocity["Median"],
                        statisticsFromPointsDistanceVelocity["Lower Quartile"],
                        statisticsFromPointsDistanceVelocity["Second Quartile"],
                        statisticsFromPointsDistanceVelocity["Third Quartile"],
                        statisticsFromPointsDistanceVelocity["Quadratic Mean"],
                        statisticsFromPointsDistanceVelocity["Harmonic Mean"],
                        statisticsFromPointsDistanceVelocity["Geometric Mean"],
                        statisticsFromPointsDistanceVelocity["Mean Absolute Deviation"],
                        statisticsFromPointsDistanceVelocity["First Point value"],
                        statisticsFromPointsDistanceVelocity["Last Point value"],
                        statisticsFromPointsDistanceVelocity["Twenty Percent"],
                        statisticsFromPointsDistanceVelocity["Eighty Percent"],
                        statisticsFromPointsDistanceVelocity["Largest Deviation Point Value"],
                        statisticsFromPointsDistanceVelocity["Time to reach Maximum Value"],
                        statisticsFromPointsDistanceVelocity["Time to reach Minimum Value"],
                        statisticsFromPointsDistanceVelocity["Length to reach Maximum Value"],
                        statisticsFromPointsDistanceVelocity["Length to reach Minimum Value"],
                        statisticsFromPointsDistanceVelocity["Length of Maximum Value to End"],
                        statisticsFromPointsDistanceVelocity["Length of Minimum Value to End"],
                        statisticsFromPointsDistanceVelocity["Skewness"],
                        statisticsFromPointsDistanceVelocity["Variance"],
                        statisticsFromPointsDistanceVelocity["Coefcient of Variation"],
                        statisticsFromPointsDistanceVelocity["Standard Error of the Mean"],
                        statisticsFromPointsDistanceVelocity["Interquartile Range"],
                        statisticsFromPointsDistanceVelocity["Median Absolute Deviation"],
                        statisticsFromPointsDistanceVelocity["Median Value at First 3 Points"],
                        statisticsFromPointsDistanceVelocity["Median Value at First 5 Points"],
                        statisticsFromPointsDistanceVelocity["Median Value at Last 3 Points"],
                        statisticsFromPointsDistanceVelocity["Median Value at Last 5 Points"],
                        statisticsFromPointsDistanceVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromPointsDistanceVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromPointsDistanceVelocity["maximum_value_portion"],
                        statisticsFromPointsDistanceVelocity["minimum_value_portion"],
                        statisticsFromPointsDistanceAcc["Kurtosis"],
                        statisticsFromPointsDistanceAcc["Mean"],
                        statisticsFromPointsDistanceAcc["Mean Before Max Deviation Point"],
                        statisticsFromPointsDistanceAcc["Mean After Max Deviation Point"],
                        statisticsFromPointsDistanceAcc["Minimum"],
                        statisticsFromPointsDistanceAcc["Maximum"],
                        statisticsFromPointsDistanceAcc["Standard Deviation"],
                        statisticsFromPointsDistanceAcc["Median"],
                        statisticsFromPointsDistanceAcc["Lower Quartile"],
                        statisticsFromPointsDistanceAcc["Second Quartile"],
                        statisticsFromPointsDistanceAcc["Third Quartile"],
                        statisticsFromPointsDistanceAcc["Quadratic Mean"],
                        statisticsFromPointsDistanceAcc["Harmonic Mean"],
                        statisticsFromPointsDistanceAcc["Geometric Mean"],
                        statisticsFromPointsDistanceAcc["Mean Absolute Deviation"],
                        statisticsFromPointsDistanceAcc["First Point value"],
                        statisticsFromPointsDistanceAcc["Last Point value"],
                        statisticsFromPointsDistanceAcc["Twenty Percent"],
                        statisticsFromPointsDistanceAcc["Eighty Percent"],
                        statisticsFromPointsDistanceAcc["Largest Deviation Point Value"],
                        statisticsFromPointsDistanceAcc["Time to reach Maximum Value"],
                        statisticsFromPointsDistanceAcc["Time to reach Minimum Value"],
                        statisticsFromPointsDistanceAcc["Length to reach Maximum Value"],
                        statisticsFromPointsDistanceAcc["Length to reach Minimum Value"],
                        statisticsFromPointsDistanceAcc["Length of Maximum Value to End"],
                        statisticsFromPointsDistanceAcc["Length of Minimum Value to End"],
                        statisticsFromPointsDistanceAcc["Skewness"],
                        statisticsFromPointsDistanceAcc["Variance"],
                        statisticsFromPointsDistanceAcc["Coefcient of Variation"],
                        statisticsFromPointsDistanceAcc["Standard Error of the Mean"],
                        statisticsFromPointsDistanceAcc["Interquartile Range"],
                        statisticsFromPointsDistanceAcc["Median Absolute Deviation"],
                        statisticsFromPointsDistanceAcc["Median Value at First 3 Points"],
                        statisticsFromPointsDistanceAcc["Median Value at First 5 Points"],
                        statisticsFromPointsDistanceAcc["Median Value at Last 3 Points"],
                        statisticsFromPointsDistanceAcc["Median Value at Last 5 Points"],
                        statisticsFromPointsDistanceAcc["Maximum Value Point to End Point Time"],
                        statisticsFromPointsDistanceAcc["Minimum Value Point to End Point Time"],
                        statisticsFromPointsDistanceAcc["maximum_value_portion"],
                        statisticsFromPointsDistanceAcc["minimum_value_portion"],
                        statisticsFromPressure["Kurtosis"],
                        statisticsFromPressure["Mean"],
                        statisticsFromPressure["Mean Before Max Deviation Point"],
                        statisticsFromPressure["Mean After Max Deviation Point"],
                        statisticsFromPressure["Minimum"],
                        statisticsFromPressure["Maximum"],
                        statisticsFromPressure["Standard Deviation"],
                        statisticsFromPressure["Median"],
                        statisticsFromPressure["Lower Quartile"],
                        statisticsFromPressure["Second Quartile"],
                        statisticsFromPressure["Third Quartile"],
                        statisticsFromPressure["Quadratic Mean"],
                        statisticsFromPressure["Harmonic Mean"],
                        statisticsFromPressure["Geometric Mean"],
                        statisticsFromPressure["Mean Absolute Deviation"],
                        statisticsFromPressure["First Point value"],
                        statisticsFromPressure["Last Point value"],
                        statisticsFromPressure["Twenty Percent"],
                        statisticsFromPressure["Eighty Percent"],
                        statisticsFromPressure["Largest Deviation Point Value"],
                        statisticsFromPressure["Time to reach Maximum Value"],
                        statisticsFromPressure["Time to reach Minimum Value"],
                        statisticsFromPressure["Length to reach Maximum Value"],
                        statisticsFromPressure["Length to reach Minimum Value"],
                        statisticsFromPressure["Length of Maximum Value to End"],
                        statisticsFromPressure["Length of Minimum Value to End"],
                        statisticsFromPressure["Skewness"],
                        statisticsFromPressure["Variance"],
                        statisticsFromPressure["Coefcient of Variation"],
                        statisticsFromPressure["Standard Error of the Mean"],
                        statisticsFromPressure["Interquartile Range"],
                        statisticsFromPressure["Median Absolute Deviation"],
                        statisticsFromPressure["Median Value at First 3 Points"],
                        statisticsFromPressure["Median Value at First 5 Points"],
                        statisticsFromPressure["Median Value at Last 3 Points"],
                        statisticsFromPressure["Median Value at Last 5 Points"],
                        statisticsFromPressure["Maximum Value Point to End Point Time"],
                        statisticsFromPressure["Minimum Value Point to End Point Time"],
                        statisticsFromPressure["maximum_value_portion"],
                        statisticsFromPressure["minimum_value_portion"],
                        statisticsFromPressureVelocity["Kurtosis"],
                        statisticsFromPressureVelocity["Mean"],
                        statisticsFromPressureVelocity["Mean Before Max Deviation Point"],
                        statisticsFromPressureVelocity["Mean After Max Deviation Point"],
                        statisticsFromPressureVelocity["Minimum"],
                        statisticsFromPressureVelocity["Maximum"],
                        statisticsFromPressureVelocity["Standard Deviation"],
                        statisticsFromPressureVelocity["Median"],
                        statisticsFromPressureVelocity["Lower Quartile"],
                        statisticsFromPressureVelocity["Second Quartile"],
                        statisticsFromPressureVelocity["Third Quartile"],
                        statisticsFromPressureVelocity["Quadratic Mean"],
                        statisticsFromPressureVelocity["Harmonic Mean"],
                        statisticsFromPressureVelocity["Geometric Mean"],
                        statisticsFromPressureVelocity["Mean Absolute Deviation"],
                        statisticsFromPressureVelocity["First Point value"],
                        statisticsFromPressureVelocity["Last Point value"],
                        statisticsFromPressureVelocity["Twenty Percent"],
                        statisticsFromPressureVelocity["Eighty Percent"],
                        statisticsFromPressureVelocity["Largest Deviation Point Value"],
                        statisticsFromPressureVelocity["Time to reach Maximum Value"],
                        statisticsFromPressureVelocity["Time to reach Minimum Value"],
                        statisticsFromPressureVelocity["Length to reach Maximum Value"],
                        statisticsFromPressureVelocity["Length to reach Minimum Value"],
                        statisticsFromPressureVelocity["Length of Maximum Value to End"],
                        statisticsFromPressureVelocity["Length of Minimum Value to End"],
                        statisticsFromPressureVelocity["Skewness"],
                        statisticsFromPressureVelocity["Variance"],
                        statisticsFromPressureVelocity["Coefcient of Variation"],
                        statisticsFromPressureVelocity["Standard Error of the Mean"],
                        statisticsFromPressureVelocity["Interquartile Range"],
                        statisticsFromPressureVelocity["Median Absolute Deviation"],
                        statisticsFromPressureVelocity["Median Value at First 3 Points"],
                        statisticsFromPressureVelocity["Median Value at First 5 Points"],
                        statisticsFromPressureVelocity["Median Value at Last 3 Points"],
                        statisticsFromPressureVelocity["Median Value at Last 5 Points"],
                        statisticsFromPressureVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromPressureVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromPressureVelocity["maximum_value_portion"],
                        statisticsFromPressureVelocity["minimum_value_portion"],
                        statisticsFromPressureAcc["Kurtosis"],
                        statisticsFromPressureAcc["Mean"],
                        statisticsFromPressureAcc["Mean Before Max Deviation Point"],
                        statisticsFromPressureAcc["Mean After Max Deviation Point"],
                        statisticsFromPressureAcc["Minimum"],
                        statisticsFromPressureAcc["Maximum"],
                        statisticsFromPressureAcc["Standard Deviation"],
                        statisticsFromPressureAcc["Median"],
                        statisticsFromPressureAcc["Lower Quartile"],
                        statisticsFromPressureAcc["Second Quartile"],
                        statisticsFromPressureAcc["Third Quartile"],
                        statisticsFromPressureAcc["Quadratic Mean"],
                        statisticsFromPressureAcc["Harmonic Mean"],
                        statisticsFromPressureAcc["Geometric Mean"],
                        statisticsFromPressureAcc["Mean Absolute Deviation"],
                        statisticsFromPressureAcc["First Point value"],
                        statisticsFromPressureAcc["Last Point value"],
                        statisticsFromPressureAcc["Twenty Percent"],
                        statisticsFromPressureAcc["Eighty Percent"],
                        statisticsFromPressureAcc["Largest Deviation Point Value"],
                        statisticsFromPressureAcc["Time to reach Maximum Value"],
                        statisticsFromPressureAcc["Time to reach Minimum Value"],
                        statisticsFromPressureAcc["Length to reach Maximum Value"],
                        statisticsFromPressureAcc["Length to reach Minimum Value"],
                        statisticsFromPressureAcc["Length of Maximum Value to End"],
                        statisticsFromPressureAcc["Length of Minimum Value to End"],
                        statisticsFromPressureAcc["Skewness"],
                        statisticsFromPressureAcc["Variance"],
                        statisticsFromPressureAcc["Coefcient of Variation"],
                        statisticsFromPressureAcc["Standard Error of the Mean"],
                        statisticsFromPressureAcc["Interquartile Range"],
                        statisticsFromPressureAcc["Median Absolute Deviation"],
                        statisticsFromPressureAcc["Median Value at First 3 Points"],
                        statisticsFromPressureAcc["Median Value at First 5 Points"],
                        statisticsFromPressureAcc["Median Value at Last 3 Points"],
                        statisticsFromPressureAcc["Median Value at Last 5 Points"],
                        statisticsFromPressureAcc["Maximum Value Point to End Point Time"],
                        statisticsFromPressureAcc["Minimum Value Point to End Point Time"],
                        statisticsFromPressureAcc["maximum_value_portion"],
                        statisticsFromPressureAcc["minimum_value_portion"],
                        statisticsFromArea["Kurtosis"],
                        statisticsFromArea["Mean"],
                        statisticsFromArea["Mean Before Max Deviation Point"],
                        statisticsFromArea["Mean After Max Deviation Point"],
                        statisticsFromArea["Minimum"],
                        statisticsFromArea["Maximum"],
                        statisticsFromArea["Standard Deviation"],
                        statisticsFromArea["Median"],
                        statisticsFromArea["Lower Quartile"],
                        statisticsFromArea["Second Quartile"],
                        statisticsFromArea["Third Quartile"],
                        statisticsFromArea["Quadratic Mean"],
                        statisticsFromArea["Harmonic Mean"],
                        statisticsFromArea["Geometric Mean"],
                        statisticsFromArea["Mean Absolute Deviation"],
                        statisticsFromArea["First Point value"],
                        statisticsFromArea["Last Point value"],
                        statisticsFromArea["Twenty Percent"],
                        statisticsFromArea["Eighty Percent"],
                        statisticsFromArea["Largest Deviation Point Value"],
                        statisticsFromArea["Time to reach Maximum Value"],
                        statisticsFromArea["Time to reach Minimum Value"],
                        statisticsFromArea["Length to reach Maximum Value"],
                        statisticsFromArea["Length to reach Minimum Value"],
                        statisticsFromArea["Length of Maximum Value to End"],
                        statisticsFromArea["Length of Minimum Value to End"],
                        statisticsFromArea["Skewness"],
                        statisticsFromArea["Variance"],
                        statisticsFromArea["Coefcient of Variation"],
                        statisticsFromArea["Standard Error of the Mean"],
                        statisticsFromArea["Interquartile Range"],
                        statisticsFromArea["Median Absolute Deviation"],
                        statisticsFromArea["Median Value at First 3 Points"],
                        statisticsFromArea["Median Value at First 5 Points"],
                        statisticsFromArea["Median Value at Last 3 Points"],
                        statisticsFromArea["Median Value at Last 5 Points"],
                        statisticsFromArea["Maximum Value Point to End Point Time"],
                        statisticsFromArea["Minimum Value Point to End Point Time"],
                        statisticsFromArea["maximum_value_portion"],
                        statisticsFromArea["minimum_value_portion"],
                        statisticsFromAreaVelocity["Kurtosis"],
                        statisticsFromAreaVelocity["Mean"],
                        statisticsFromAreaVelocity["Mean Before Max Deviation Point"],
                        statisticsFromAreaVelocity["Mean After Max Deviation Point"],
                        statisticsFromAreaVelocity["Minimum"],
                        statisticsFromAreaVelocity["Maximum"],
                        statisticsFromAreaVelocity["Standard Deviation"],
                        statisticsFromAreaVelocity["Median"],
                        statisticsFromAreaVelocity["Lower Quartile"],
                        statisticsFromAreaVelocity["Second Quartile"],
                        statisticsFromAreaVelocity["Third Quartile"],
                        statisticsFromAreaVelocity["Quadratic Mean"],
                        statisticsFromAreaVelocity["Harmonic Mean"],
                        statisticsFromAreaVelocity["Geometric Mean"],
                        statisticsFromAreaVelocity["Mean Absolute Deviation"],
                        statisticsFromAreaVelocity["First Point value"],
                        statisticsFromAreaVelocity["Last Point value"],
                        statisticsFromAreaVelocity["Twenty Percent"],
                        statisticsFromAreaVelocity["Eighty Percent"],
                        statisticsFromAreaVelocity["Largest Deviation Point Value"],
                        statisticsFromAreaVelocity["Time to reach Maximum Value"],
                        statisticsFromAreaVelocity["Time to reach Minimum Value"],
                        statisticsFromAreaVelocity["Length to reach Maximum Value"],
                        statisticsFromAreaVelocity["Length to reach Minimum Value"],
                        statisticsFromAreaVelocity["Length of Maximum Value to End"],
                        statisticsFromAreaVelocity["Length of Minimum Value to End"],
                        statisticsFromAreaVelocity["Skewness"],
                        statisticsFromAreaVelocity["Variance"],
                        statisticsFromAreaVelocity["Coefcient of Variation"],
                        statisticsFromAreaVelocity["Standard Error of the Mean"],
                        statisticsFromAreaVelocity["Interquartile Range"],
                        statisticsFromAreaVelocity["Median Absolute Deviation"],
                        statisticsFromAreaVelocity["Median Value at First 3 Points"],
                        statisticsFromAreaVelocity["Median Value at First 5 Points"],
                        statisticsFromAreaVelocity["Median Value at Last 3 Points"],
                        statisticsFromAreaVelocity["Median Value at Last 5 Points"],
                        statisticsFromAreaVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromAreaVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromAreaVelocity["maximum_value_portion"],
                        statisticsFromAreaVelocity["minimum_value_portion"],
                        statisticsFromAreaAcc["Kurtosis"],
                        statisticsFromAreaAcc["Mean"],
                        statisticsFromAreaAcc["Mean Before Max Deviation Point"],
                        statisticsFromAreaAcc["Mean After Max Deviation Point"],
                        statisticsFromAreaAcc["Minimum"],
                        statisticsFromAreaAcc["Maximum"],
                        statisticsFromAreaAcc["Standard Deviation"],
                        statisticsFromAreaAcc["Median"],
                        statisticsFromAreaAcc["Lower Quartile"],
                        statisticsFromAreaAcc["Second Quartile"],
                        statisticsFromAreaAcc["Third Quartile"],
                        statisticsFromAreaAcc["Quadratic Mean"],
                        statisticsFromAreaAcc["Harmonic Mean"],
                        statisticsFromAreaAcc["Geometric Mean"],
                        statisticsFromAreaAcc["Mean Absolute Deviation"],
                        statisticsFromAreaAcc["First Point value"],
                        statisticsFromAreaAcc["Last Point value"],
                        statisticsFromAreaAcc["Twenty Percent"],
                        statisticsFromAreaAcc["Eighty Percent"],
                        statisticsFromAreaAcc["Largest Deviation Point Value"],
                        statisticsFromAreaAcc["Time to reach Maximum Value"],
                        statisticsFromAreaAcc["Time to reach Minimum Value"],
                        statisticsFromAreaAcc["Length to reach Maximum Value"],
                        statisticsFromAreaAcc["Length to reach Minimum Value"],
                        statisticsFromAreaAcc["Length of Maximum Value to End"],
                        statisticsFromAreaAcc["Length of Minimum Value to End"],
                        statisticsFromAreaAcc["Skewness"],
                        statisticsFromAreaAcc["Variance"],
                        statisticsFromAreaAcc["Coefcient of Variation"],
                        statisticsFromAreaAcc["Standard Error of the Mean"],
                        statisticsFromAreaAcc["Interquartile Range"],
                        statisticsFromAreaAcc["Median Absolute Deviation"],
                        statisticsFromAreaAcc["Median Value at First 3 Points"],
                        statisticsFromAreaAcc["Median Value at First 5 Points"],
                        statisticsFromAreaAcc["Median Value at Last 3 Points"],
                        statisticsFromAreaAcc["Median Value at Last 5 Points"],
                        statisticsFromAreaAcc["Maximum Value Point to End Point Time"],
                        statisticsFromAreaAcc["Minimum Value Point to End Point Time"],
                        statisticsFromAreaAcc["maximum_value_portion"],
                        statisticsFromAreaAcc["minimum_value_portion"],
                        statisticsFromAngl["Kurtosis"],
                        statisticsFromAngl["Mean"],
                        statisticsFromAngl["Mean Before Max Deviation Point"],
                        statisticsFromAngl["Mean After Max Deviation Point"],
                        statisticsFromAngl["Minimum"],
                        statisticsFromAngl["Maximum"],
                        statisticsFromAngl["Standard Deviation"],
                        statisticsFromAngl["Median"],
                        statisticsFromAngl["Lower Quartile"],
                        statisticsFromAngl["Second Quartile"],
                        statisticsFromAngl["Third Quartile"],
                        statisticsFromAngl["Quadratic Mean"],
                        statisticsFromAngl["Harmonic Mean"],
                        statisticsFromAngl["Geometric Mean"],
                        statisticsFromAngl["Mean Absolute Deviation"],
                        statisticsFromAngl["First Point value"],
                        statisticsFromAngl["Last Point value"],
                        statisticsFromAngl["Twenty Percent"],
                        statisticsFromAngl["Eighty Percent"],
                        statisticsFromAngl["Largest Deviation Point Value"],
                        statisticsFromAngl["Time to reach Maximum Value"],
                        statisticsFromAngl["Time to reach Minimum Value"],
                        statisticsFromAngl["Length to reach Maximum Value"],
                        statisticsFromAngl["Length to reach Minimum Value"],
                        statisticsFromAngl["Length of Maximum Value to End"],
                        statisticsFromAngl["Length of Minimum Value to End"],
                        statisticsFromAngl["Skewness"],
                        statisticsFromAngl["Variance"],
                        statisticsFromAngl["Coefcient of Variation"],
                        statisticsFromAngl["Standard Error of the Mean"],
                        statisticsFromAngl["Interquartile Range"],
                        statisticsFromAngl["Median Absolute Deviation"],
                        statisticsFromAngl["Median Value at First 3 Points"],
                        statisticsFromAngl["Median Value at First 5 Points"],
                        statisticsFromAngl["Median Value at Last 3 Points"],
                        statisticsFromAngl["Median Value at Last 5 Points"],
                        statisticsFromAngl["Maximum Value Point to End Point Time"],
                        statisticsFromAngl["Minimum Value Point to End Point Time"],
                        statisticsFromAngl["maximum_value_portion"],
                        statisticsFromAngl["minimum_value_portion"],
                        statisticsFromAnglVelocity["Kurtosis"],
                        statisticsFromAnglVelocity["Mean"],
                        statisticsFromAnglVelocity["Mean Before Max Deviation Point"],
                        statisticsFromAnglVelocity["Mean After Max Deviation Point"],
                        statisticsFromAnglVelocity["Minimum"],
                        statisticsFromAnglVelocity["Maximum"],
                        statisticsFromAnglVelocity["Standard Deviation"],
                        statisticsFromAnglVelocity["Median"],
                        statisticsFromAnglVelocity["Lower Quartile"],
                        statisticsFromAnglVelocity["Second Quartile"],
                        statisticsFromAnglVelocity["Third Quartile"],
                        statisticsFromAnglVelocity["Quadratic Mean"],
                        statisticsFromAnglVelocity["Harmonic Mean"],
                        statisticsFromAnglVelocity["Geometric Mean"],
                        statisticsFromAnglVelocity["Mean Absolute Deviation"],
                        statisticsFromAnglVelocity["First Point value"],
                        statisticsFromAnglVelocity["Last Point value"],
                        statisticsFromAnglVelocity["Twenty Percent"],
                        statisticsFromAnglVelocity["Eighty Percent"],
                        statisticsFromAnglVelocity["Largest Deviation Point Value"],
                        statisticsFromAnglVelocity["Time to reach Maximum Value"],
                        statisticsFromAnglVelocity["Time to reach Minimum Value"],
                        statisticsFromAnglVelocity["Length to reach Maximum Value"],
                        statisticsFromAnglVelocity["Length to reach Minimum Value"],
                        statisticsFromAnglVelocity["Length of Maximum Value to End"],
                        statisticsFromAnglVelocity["Length of Minimum Value to End"],
                        statisticsFromAnglVelocity["Skewness"],
                        statisticsFromAnglVelocity["Variance"],
                        statisticsFromAnglVelocity["Coefcient of Variation"],
                        statisticsFromAnglVelocity["Standard Error of the Mean"],
                        statisticsFromAnglVelocity["Interquartile Range"],
                        statisticsFromAnglVelocity["Median Absolute Deviation"],
                        statisticsFromAnglVelocity["Median Value at First 3 Points"],
                        statisticsFromAnglVelocity["Median Value at First 5 Points"],
                        statisticsFromAnglVelocity["Median Value at Last 3 Points"],
                        statisticsFromAnglVelocity["Median Value at Last 5 Points"],
                        statisticsFromAnglVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromAnglVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromAnglVelocity["maximum_value_portion"],
                        statisticsFromAnglVelocity["minimum_value_portion"],
                        statisticsFromAnglAcc["Kurtosis"],
                        statisticsFromAnglAcc["Mean"],
                        statisticsFromAnglAcc["Mean Before Max Deviation Point"],
                        statisticsFromAnglAcc["Mean After Max Deviation Point"],
                        statisticsFromAnglAcc["Minimum"],
                        statisticsFromAnglAcc["Maximum"],
                        statisticsFromAnglAcc["Standard Deviation"],
                        statisticsFromAnglAcc["Median"],
                        statisticsFromAnglAcc["Lower Quartile"],
                        statisticsFromAnglAcc["Second Quartile"],
                        statisticsFromAnglAcc["Third Quartile"],
                        statisticsFromAnglAcc["Quadratic Mean"],
                        statisticsFromAnglAcc["Harmonic Mean"],
                        statisticsFromAnglAcc["Geometric Mean"],
                        statisticsFromAnglAcc["Mean Absolute Deviation"],
                        statisticsFromAnglAcc["First Point value"],
                        statisticsFromAnglAcc["Last Point value"],
                        statisticsFromAnglAcc["Twenty Percent"],
                        statisticsFromAnglAcc["Eighty Percent"],
                        statisticsFromAnglAcc["Largest Deviation Point Value"],
                        statisticsFromAnglAcc["Time to reach Maximum Value"],
                        statisticsFromAnglAcc["Time to reach Minimum Value"],
                        statisticsFromAnglAcc["Length to reach Maximum Value"],
                        statisticsFromAnglAcc["Length to reach Minimum Value"],
                        statisticsFromAnglAcc["Length of Maximum Value to End"],
                        statisticsFromAnglAcc["Length of Minimum Value to End"],
                        statisticsFromAnglAcc["Skewness"],
                        statisticsFromAnglAcc["Variance"],
                        statisticsFromAnglAcc["Coefcient of Variation"],
                        statisticsFromAnglAcc["Standard Error of the Mean"],
                        statisticsFromAnglAcc["Interquartile Range"],
                        statisticsFromAnglAcc["Median Absolute Deviation"],
                        statisticsFromAnglAcc["Median Value at First 3 Points"],
                        statisticsFromAnglAcc["Median Value at First 5 Points"],
                        statisticsFromAnglAcc["Median Value at Last 3 Points"],
                        statisticsFromAnglAcc["Median Value at Last 5 Points"],
                        statisticsFromAnglAcc["Maximum Value Point to End Point Time"],
                        statisticsFromAnglAcc["Minimum Value Point to End Point Time"],
                        statisticsFromAnglAcc["maximum_value_portion"],
                        statisticsFromAnglAcc["minimum_value_portion"],
                        statisticsFromPairwDist["Kurtosis"],
                        statisticsFromPairwDist["Mean"],
                        statisticsFromPairwDist["Mean Before Max Deviation Point"],
                        statisticsFromPairwDist["Mean After Max Deviation Point"],
                        statisticsFromPairwDist["Minimum"],
                        statisticsFromPairwDist["Maximum"],
                        statisticsFromPairwDist["Standard Deviation"],
                        statisticsFromPairwDist["Median"],
                        statisticsFromPairwDist["Lower Quartile"],
                        statisticsFromPairwDist["Second Quartile"],
                        statisticsFromPairwDist["Third Quartile"],
                        statisticsFromPairwDist["Quadratic Mean"],
                        statisticsFromPairwDist["Harmonic Mean"],
                        statisticsFromPairwDist["Geometric Mean"],
                        statisticsFromPairwDist["Mean Absolute Deviation"],
                        statisticsFromPairwDist["First Point value"],
                        statisticsFromPairwDist["Last Point value"],
                        statisticsFromPairwDist["Twenty Percent"],
                        statisticsFromPairwDist["Eighty Percent"],
                        statisticsFromPairwDist["Largest Deviation Point Value"],
                        statisticsFromPairwDist["Time to reach Maximum Value"],
                        statisticsFromPairwDist["Time to reach Minimum Value"],
                        statisticsFromPairwDist["Length to reach Maximum Value"],
                        statisticsFromPairwDist["Length to reach Minimum Value"],
                        statisticsFromPairwDist["Length of Maximum Value to End"],
                        statisticsFromPairwDist["Length of Minimum Value to End"],
                        statisticsFromPairwDist["Skewness"],
                        statisticsFromPairwDist["Variance"],
                        statisticsFromPairwDist["Coefcient of Variation"],
                        statisticsFromPairwDist["Standard Error of the Mean"],
                        statisticsFromPairwDist["Interquartile Range"],
                        statisticsFromPairwDist["Median Absolute Deviation"],
                        statisticsFromPairwDist["Median Value at First 3 Points"],
                        statisticsFromPairwDist["Median Value at First 5 Points"],
                        statisticsFromPairwDist["Median Value at Last 3 Points"],
                        statisticsFromPairwDist["Median Value at Last 5 Points"],
                        statisticsFromPairwDist["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDist["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDist["maximum_value_portion"],
                        statisticsFromPairwDist["minimum_value_portion"],
                        statisticsFromPairwDistX["Kurtosis"],
                        statisticsFromPairwDistX["Mean"],
                        statisticsFromPairwDistX["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistX["Mean After Max Deviation Point"],
                        statisticsFromPairwDistX["Minimum"],
                        statisticsFromPairwDistX["Maximum"],
                        statisticsFromPairwDistX["Standard Deviation"],
                        statisticsFromPairwDistX["Median"],
                        statisticsFromPairwDistX["Lower Quartile"],
                        statisticsFromPairwDistX["Second Quartile"],
                        statisticsFromPairwDistX["Third Quartile"],
                        statisticsFromPairwDistX["Quadratic Mean"],
                        statisticsFromPairwDistX["Harmonic Mean"],
                        statisticsFromPairwDistX["Geometric Mean"],
                        statisticsFromPairwDistX["Mean Absolute Deviation"],
                        statisticsFromPairwDistX["First Point value"],
                        statisticsFromPairwDistX["Last Point value"],
                        statisticsFromPairwDistX["Twenty Percent"],
                        statisticsFromPairwDistX["Eighty Percent"],
                        statisticsFromPairwDistX["Largest Deviation Point Value"],
                        statisticsFromPairwDistX["Time to reach Maximum Value"],
                        statisticsFromPairwDistX["Time to reach Minimum Value"],
                        statisticsFromPairwDistX["Length to reach Maximum Value"],
                        statisticsFromPairwDistX["Length to reach Minimum Value"],
                        statisticsFromPairwDistX["Length of Maximum Value to End"],
                        statisticsFromPairwDistX["Length of Minimum Value to End"],
                        statisticsFromPairwDistX["Skewness"],
                        statisticsFromPairwDistX["Variance"],
                        statisticsFromPairwDistX["Coefcient of Variation"],
                        statisticsFromPairwDistX["Standard Error of the Mean"],
                        statisticsFromPairwDistX["Interquartile Range"],
                        statisticsFromPairwDistX["Median Absolute Deviation"],
                        statisticsFromPairwDistX["Median Value at First 3 Points"],
                        statisticsFromPairwDistX["Median Value at First 5 Points"],
                        statisticsFromPairwDistX["Median Value at Last 3 Points"],
                        statisticsFromPairwDistX["Median Value at Last 5 Points"],
                        statisticsFromPairwDistX["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistX["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistX["maximum_value_portion"],
                        statisticsFromPairwDistX["minimum_value_portion"],
                        statisticsFromPairwDistY["Kurtosis"],
                        statisticsFromPairwDistY["Mean"],
                        statisticsFromPairwDistY["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistY["Mean After Max Deviation Point"],
                        statisticsFromPairwDistY["Minimum"],
                        statisticsFromPairwDistY["Maximum"],
                        statisticsFromPairwDistY["Standard Deviation"],
                        statisticsFromPairwDistY["Median"],
                        statisticsFromPairwDistY["Lower Quartile"],
                        statisticsFromPairwDistY["Second Quartile"],
                        statisticsFromPairwDistY["Third Quartile"],
                        statisticsFromPairwDistY["Quadratic Mean"],
                        statisticsFromPairwDistY["Harmonic Mean"],
                        statisticsFromPairwDistY["Geometric Mean"],
                        statisticsFromPairwDistY["Mean Absolute Deviation"],
                        statisticsFromPairwDistY["First Point value"],
                        statisticsFromPairwDistY["Last Point value"],
                        statisticsFromPairwDistY["Twenty Percent"],
                        statisticsFromPairwDistY["Eighty Percent"],
                        statisticsFromPairwDistY["Largest Deviation Point Value"],
                        statisticsFromPairwDistY["Time to reach Maximum Value"],
                        statisticsFromPairwDistY["Time to reach Minimum Value"],
                        statisticsFromPairwDistY["Length to reach Maximum Value"],
                        statisticsFromPairwDistY["Length to reach Minimum Value"],
                        statisticsFromPairwDistY["Length of Maximum Value to End"],
                        statisticsFromPairwDistY["Length of Minimum Value to End"],
                        statisticsFromPairwDistY["Skewness"],
                        statisticsFromPairwDistY["Variance"],
                        statisticsFromPairwDistY["Coefcient of Variation"],
                        statisticsFromPairwDistY["Standard Error of the Mean"],
                        statisticsFromPairwDistY["Interquartile Range"],
                        statisticsFromPairwDistY["Median Absolute Deviation"],
                        statisticsFromPairwDistY["Median Value at First 3 Points"],
                        statisticsFromPairwDistY["Median Value at First 5 Points"],
                        statisticsFromPairwDistY["Median Value at Last 3 Points"],
                        statisticsFromPairwDistY["Median Value at Last 5 Points"],
                        statisticsFromPairwDistY["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistY["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistY["maximum_value_portion"],
                        statisticsFromPairwDistY["minimum_value_portion"],
                        statisticsFromPairwDistVelocity["Kurtosis"],
                        statisticsFromPairwDistVelocity["Mean"],
                        statisticsFromPairwDistVelocity["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistVelocity["Mean After Max Deviation Point"],
                        statisticsFromPairwDistVelocity["Minimum"],
                        statisticsFromPairwDistVelocity["Maximum"],
                        statisticsFromPairwDistVelocity["Standard Deviation"],
                        statisticsFromPairwDistVelocity["Median"],
                        statisticsFromPairwDistVelocity["Lower Quartile"],
                        statisticsFromPairwDistVelocity["Second Quartile"],
                        statisticsFromPairwDistVelocity["Third Quartile"],
                        statisticsFromPairwDistVelocity["Quadratic Mean"],
                        statisticsFromPairwDistVelocity["Harmonic Mean"],
                        statisticsFromPairwDistVelocity["Geometric Mean"],
                        statisticsFromPairwDistVelocity["Mean Absolute Deviation"],
                        statisticsFromPairwDistVelocity["First Point value"],
                        statisticsFromPairwDistVelocity["Last Point value"],
                        statisticsFromPairwDistVelocity["Twenty Percent"],
                        statisticsFromPairwDistVelocity["Eighty Percent"],
                        statisticsFromPairwDistVelocity["Largest Deviation Point Value"],
                        statisticsFromPairwDistVelocity["Time to reach Maximum Value"],
                        statisticsFromPairwDistVelocity["Time to reach Minimum Value"],
                        statisticsFromPairwDistVelocity["Length to reach Maximum Value"],
                        statisticsFromPairwDistVelocity["Length to reach Minimum Value"],
                        statisticsFromPairwDistVelocity["Length of Maximum Value to End"],
                        statisticsFromPairwDistVelocity["Length of Minimum Value to End"],
                        statisticsFromPairwDistVelocity["Skewness"],
                        statisticsFromPairwDistVelocity["Variance"],
                        statisticsFromPairwDistVelocity["Coefcient of Variation"],
                        statisticsFromPairwDistVelocity["Standard Error of the Mean"],
                        statisticsFromPairwDistVelocity["Interquartile Range"],
                        statisticsFromPairwDistVelocity["Median Absolute Deviation"],
                        statisticsFromPairwDistVelocity["Median Value at First 3 Points"],
                        statisticsFromPairwDistVelocity["Median Value at First 5 Points"],
                        statisticsFromPairwDistVelocity["Median Value at Last 3 Points"],
                        statisticsFromPairwDistVelocity["Median Value at Last 5 Points"],
                        statisticsFromPairwDistVelocity["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocity["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocity["maximum_value_portion"],
                        statisticsFromPairwDistVelocity["minimum_value_portion"],
                        statisticsFromPairwDistVelocityX["Kurtosis"],
                        statisticsFromPairwDistVelocityX["Mean"],
                        statisticsFromPairwDistVelocityX["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistVelocityX["Mean After Max Deviation Point"],
                        statisticsFromPairwDistVelocityX["Minimum"],
                        statisticsFromPairwDistVelocityX["Maximum"],
                        statisticsFromPairwDistVelocityX["Standard Deviation"],
                        statisticsFromPairwDistVelocityX["Median"],
                        statisticsFromPairwDistVelocityX["Lower Quartile"],
                        statisticsFromPairwDistVelocityX["Second Quartile"],
                        statisticsFromPairwDistVelocityX["Third Quartile"],
                        statisticsFromPairwDistVelocityX["Quadratic Mean"],
                        statisticsFromPairwDistVelocityX["Harmonic Mean"],
                        statisticsFromPairwDistVelocityX["Geometric Mean"],
                        statisticsFromPairwDistVelocityX["Mean Absolute Deviation"],
                        statisticsFromPairwDistVelocityX["First Point value"],
                        statisticsFromPairwDistVelocityX["Last Point value"],
                        statisticsFromPairwDistVelocityX["Twenty Percent"],
                        statisticsFromPairwDistVelocityX["Eighty Percent"],
                        statisticsFromPairwDistVelocityX["Largest Deviation Point Value"],
                        statisticsFromPairwDistVelocityX["Time to reach Maximum Value"],
                        statisticsFromPairwDistVelocityX["Time to reach Minimum Value"],
                        statisticsFromPairwDistVelocityX["Length to reach Maximum Value"],
                        statisticsFromPairwDistVelocityX["Length to reach Minimum Value"],
                        statisticsFromPairwDistVelocityX["Length of Maximum Value to End"],
                        statisticsFromPairwDistVelocityX["Length of Minimum Value to End"],
                        statisticsFromPairwDistVelocityX["Skewness"],
                        statisticsFromPairwDistVelocityX["Variance"],
                        statisticsFromPairwDistVelocityX["Coefcient of Variation"],
                        statisticsFromPairwDistVelocityX["Standard Error of the Mean"],
                        statisticsFromPairwDistVelocityX["Interquartile Range"],
                        statisticsFromPairwDistVelocityX["Median Absolute Deviation"],
                        statisticsFromPairwDistVelocityX["Median Value at First 3 Points"],
                        statisticsFromPairwDistVelocityX["Median Value at First 5 Points"],
                        statisticsFromPairwDistVelocityX["Median Value at Last 3 Points"],
                        statisticsFromPairwDistVelocityX["Median Value at Last 5 Points"],
                        statisticsFromPairwDistVelocityX["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocityX["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocityX["maximum_value_portion"],
                        statisticsFromPairwDistVelocityX["minimum_value_portion"],
                        statisticsFromPairwDistVelocityY["Kurtosis"],
                        statisticsFromPairwDistVelocityY["Mean"],
                        statisticsFromPairwDistVelocityY["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistVelocityY["Mean After Max Deviation Point"],
                        statisticsFromPairwDistVelocityY["Minimum"],
                        statisticsFromPairwDistVelocityY["Maximum"],
                        statisticsFromPairwDistVelocityY["Standard Deviation"],
                        statisticsFromPairwDistVelocityY["Median"],
                        statisticsFromPairwDistVelocityY["Lower Quartile"],
                        statisticsFromPairwDistVelocityY["Second Quartile"],
                        statisticsFromPairwDistVelocityY["Third Quartile"],
                        statisticsFromPairwDistVelocityY["Quadratic Mean"],
                        statisticsFromPairwDistVelocityY["Harmonic Mean"],
                        statisticsFromPairwDistVelocityY["Geometric Mean"],
                        statisticsFromPairwDistVelocityY["Mean Absolute Deviation"],
                        statisticsFromPairwDistVelocityY["First Point value"],
                        statisticsFromPairwDistVelocityY["Last Point value"],
                        statisticsFromPairwDistVelocityY["Twenty Percent"],
                        statisticsFromPairwDistVelocityY["Eighty Percent"],
                        statisticsFromPairwDistVelocityY["Largest Deviation Point Value"],
                        statisticsFromPairwDistVelocityY["Time to reach Maximum Value"],
                        statisticsFromPairwDistVelocityY["Time to reach Minimum Value"],
                        statisticsFromPairwDistVelocityY["Length to reach Maximum Value"],
                        statisticsFromPairwDistVelocityY["Length to reach Minimum Value"],
                        statisticsFromPairwDistVelocityY["Length of Maximum Value to End"],
                        statisticsFromPairwDistVelocityY["Length of Minimum Value to End"],
                        statisticsFromPairwDistVelocityY["Skewness"],
                        statisticsFromPairwDistVelocityY["Variance"],
                        statisticsFromPairwDistVelocityY["Coefcient of Variation"],
                        statisticsFromPairwDistVelocityY["Standard Error of the Mean"],
                        statisticsFromPairwDistVelocityY["Interquartile Range"],
                        statisticsFromPairwDistVelocityY["Median Absolute Deviation"],
                        statisticsFromPairwDistVelocityY["Median Value at First 3 Points"],
                        statisticsFromPairwDistVelocityY["Median Value at First 5 Points"],
                        statisticsFromPairwDistVelocityY["Median Value at Last 3 Points"],
                        statisticsFromPairwDistVelocityY["Median Value at Last 5 Points"],
                        statisticsFromPairwDistVelocityY["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocityY["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistVelocityY["maximum_value_portion"],
                        statisticsFromPairwDistVelocityY["minimum_value_portion"],
                        statisticsFromPairwDistAcc["Kurtosis"],
                        statisticsFromPairwDistAcc["Mean"],
                        statisticsFromPairwDistAcc["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistAcc["Mean After Max Deviation Point"],
                        statisticsFromPairwDistAcc["Minimum"],
                        statisticsFromPairwDistAcc["Maximum"],
                        statisticsFromPairwDistAcc["Standard Deviation"],
                        statisticsFromPairwDistAcc["Median"],
                        statisticsFromPairwDistAcc["Lower Quartile"],
                        statisticsFromPairwDistAcc["Second Quartile"],
                        statisticsFromPairwDistAcc["Third Quartile"],
                        statisticsFromPairwDistAcc["Quadratic Mean"],
                        statisticsFromPairwDistAcc["Harmonic Mean"],
                        statisticsFromPairwDistAcc["Geometric Mean"],
                        statisticsFromPairwDistAcc["Mean Absolute Deviation"],
                        statisticsFromPairwDistAcc["First Point value"],
                        statisticsFromPairwDistAcc["Last Point value"],
                        statisticsFromPairwDistAcc["Twenty Percent"],
                        statisticsFromPairwDistAcc["Eighty Percent"],
                        statisticsFromPairwDistAcc["Largest Deviation Point Value"],
                        statisticsFromPairwDistAcc["Time to reach Maximum Value"],
                        statisticsFromPairwDistAcc["Time to reach Minimum Value"],
                        statisticsFromPairwDistAcc["Length to reach Maximum Value"],
                        statisticsFromPairwDistAcc["Length to reach Minimum Value"],
                        statisticsFromPairwDistAcc["Length of Maximum Value to End"],
                        statisticsFromPairwDistAcc["Length of Minimum Value to End"],
                        statisticsFromPairwDistAcc["Skewness"],
                        statisticsFromPairwDistAcc["Variance"],
                        statisticsFromPairwDistAcc["Coefcient of Variation"],
                        statisticsFromPairwDistAcc["Standard Error of the Mean"],
                        statisticsFromPairwDistAcc["Interquartile Range"],
                        statisticsFromPairwDistAcc["Median Absolute Deviation"],
                        statisticsFromPairwDistAcc["Median Value at First 3 Points"],
                        statisticsFromPairwDistAcc["Median Value at First 5 Points"],
                        statisticsFromPairwDistAcc["Median Value at Last 3 Points"],
                        statisticsFromPairwDistAcc["Median Value at Last 5 Points"],
                        statisticsFromPairwDistAcc["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistAcc["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistAcc["maximum_value_portion"],
                        statisticsFromPairwDistAcc["minimum_value_portion"],
                        statisticsFromPairwDistAccX["Kurtosis"],
                        statisticsFromPairwDistAccX["Mean"],
                        statisticsFromPairwDistAccX["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistAccX["Mean After Max Deviation Point"],
                        statisticsFromPairwDistAccX["Minimum"],
                        statisticsFromPairwDistAccX["Maximum"],
                        statisticsFromPairwDistAccX["Standard Deviation"],
                        statisticsFromPairwDistAccX["Median"],
                        statisticsFromPairwDistAccX["Lower Quartile"],
                        statisticsFromPairwDistAccX["Second Quartile"],
                        statisticsFromPairwDistAccX["Third Quartile"],
                        statisticsFromPairwDistAccX["Quadratic Mean"],
                        statisticsFromPairwDistAccX["Harmonic Mean"],
                        statisticsFromPairwDistAccX["Geometric Mean"],
                        statisticsFromPairwDistAccX["Mean Absolute Deviation"],
                        statisticsFromPairwDistAccX["First Point value"],
                        statisticsFromPairwDistAccX["Last Point value"],
                        statisticsFromPairwDistAccX["Twenty Percent"],
                        statisticsFromPairwDistAccX["Eighty Percent"],
                        statisticsFromPairwDistAccX["Largest Deviation Point Value"],
                        statisticsFromPairwDistAccX["Time to reach Maximum Value"],
                        statisticsFromPairwDistAccX["Time to reach Minimum Value"],
                        statisticsFromPairwDistAccX["Length to reach Maximum Value"],
                        statisticsFromPairwDistAccX["Length to reach Minimum Value"],
                        statisticsFromPairwDistAccX["Length of Maximum Value to End"],
                        statisticsFromPairwDistAccX["Length of Minimum Value to End"],
                        statisticsFromPairwDistAccX["Skewness"],
                        statisticsFromPairwDistAccX["Variance"],
                        statisticsFromPairwDistAccX["Coefcient of Variation"],
                        statisticsFromPairwDistAccX["Standard Error of the Mean"],
                        statisticsFromPairwDistAccX["Interquartile Range"],
                        statisticsFromPairwDistAccX["Median Absolute Deviation"],
                        statisticsFromPairwDistAccX["Median Value at First 3 Points"],
                        statisticsFromPairwDistAccX["Median Value at First 5 Points"],
                        statisticsFromPairwDistAccX["Median Value at Last 3 Points"],
                        statisticsFromPairwDistAccX["Median Value at Last 5 Points"],
                        statisticsFromPairwDistAccX["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistAccX["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistAccX["maximum_value_portion"],
                        statisticsFromPairwDistAccX["minimum_value_portion"],
                        statisticsFromPairwDistAccY["Kurtosis"],
                        statisticsFromPairwDistAccY["Mean"],
                        statisticsFromPairwDistAccY["Mean Before Max Deviation Point"],
                        statisticsFromPairwDistAccY["Mean After Max Deviation Point"],
                        statisticsFromPairwDistAccY["Minimum"],
                        statisticsFromPairwDistAccY["Maximum"],
                        statisticsFromPairwDistAccY["Standard Deviation"],
                        statisticsFromPairwDistAccY["Median"],
                        statisticsFromPairwDistAccY["Lower Quartile"],
                        statisticsFromPairwDistAccY["Second Quartile"],
                        statisticsFromPairwDistAccY["Third Quartile"],
                        statisticsFromPairwDistAccY["Quadratic Mean"],
                        statisticsFromPairwDistAccY["Harmonic Mean"],
                        statisticsFromPairwDistAccY["Geometric Mean"],
                        statisticsFromPairwDistAccY["Mean Absolute Deviation"],
                        statisticsFromPairwDistAccY["First Point value"],
                        statisticsFromPairwDistAccY["Last Point value"],
                        statisticsFromPairwDistAccY["Twenty Percent"],
                        statisticsFromPairwDistAccY["Eighty Percent"],
                        statisticsFromPairwDistAccY["Largest Deviation Point Value"],
                        statisticsFromPairwDistAccY["Time to reach Maximum Value"],
                        statisticsFromPairwDistAccY["Time to reach Minimum Value"],
                        statisticsFromPairwDistAccY["Length to reach Maximum Value"],
                        statisticsFromPairwDistAccY["Length to reach Minimum Value"],
                        statisticsFromPairwDistAccY["Length of Maximum Value to End"],
                        statisticsFromPairwDistAccY["Length of Minimum Value to End"],
                        statisticsFromPairwDistAccY["Skewness"],
                        statisticsFromPairwDistAccY["Variance"],
                        statisticsFromPairwDistAccY["Coefcient of Variation"],
                        statisticsFromPairwDistAccY["Standard Error of the Mean"],
                        statisticsFromPairwDistAccY["Interquartile Range"],
                        statisticsFromPairwDistAccY["Median Absolute Deviation"],
                        statisticsFromPairwDistAccY["Median Value at First 3 Points"],
                        statisticsFromPairwDistAccY["Median Value at First 5 Points"],
                        statisticsFromPairwDistAccY["Median Value at Last 3 Points"],
                        statisticsFromPairwDistAccY["Median Value at Last 5 Points"],
                        statisticsFromPairwDistAccY["Maximum Value Point to End Point Time"],
                        statisticsFromPairwDistAccY["Minimum Value Point to End Point Time"],
                        statisticsFromPairwDistAccY["maximum_value_portion"],
                        statisticsFromPairwDistAccY["minimum_value_portion"],
                        statisticsFromDeviationSequences["Kurtosis"],
                        statisticsFromDeviationSequences["Mean"],
                        statisticsFromDeviationSequences["Mean Before Max Deviation Point"],
                        statisticsFromDeviationSequences["Mean After Max Deviation Point"],
                        statisticsFromDeviationSequences["Minimum"],
                        statisticsFromDeviationSequences["Maximum"],
                        statisticsFromDeviationSequences["Standard Deviation"],
                        statisticsFromDeviationSequences["Median"],
                        statisticsFromDeviationSequences["Lower Quartile"],
                        statisticsFromDeviationSequences["Second Quartile"],
                        statisticsFromDeviationSequences["Third Quartile"],
                        statisticsFromDeviationSequences["Quadratic Mean"],
                        statisticsFromDeviationSequences["Harmonic Mean"],
                        statisticsFromDeviationSequences["Geometric Mean"],
                        statisticsFromDeviationSequences["Mean Absolute Deviation"],
                        statisticsFromDeviationSequences["First Point value"],
                        statisticsFromDeviationSequences["Last Point value"],
                        statisticsFromDeviationSequences["Twenty Percent"],
                        statisticsFromDeviationSequences["Eighty Percent"],
                        statisticsFromDeviationSequences["Largest Deviation Point Value"],
                        statisticsFromDeviationSequences["Time to reach Maximum Value"],
                        statisticsFromDeviationSequences["Time to reach Minimum Value"],
                        statisticsFromDeviationSequences["Length to reach Maximum Value"],
                        statisticsFromDeviationSequences["Length to reach Minimum Value"],
                        statisticsFromDeviationSequences["Length of Maximum Value to End"],
                        statisticsFromDeviationSequences["Length of Minimum Value to End"],
                        statisticsFromDeviationSequences["Skewness"],
                        statisticsFromDeviationSequences["Variance"],
                        statisticsFromDeviationSequences["Coefcient of Variation"],
                        statisticsFromDeviationSequences["Standard Error of the Mean"],
                        statisticsFromDeviationSequences["Interquartile Range"],
                        statisticsFromDeviationSequences["Median Absolute Deviation"],
                        statisticsFromDeviationSequences["Median Value at First 3 Points"],
                        statisticsFromDeviationSequences["Median Value at First 5 Points"],
                        statisticsFromDeviationSequences["Median Value at Last 3 Points"],
                        statisticsFromDeviationSequences["Median Value at Last 5 Points"],
                        statisticsFromDeviationSequences["Maximum Value Point to End Point Time"],
                        statisticsFromDeviationSequences["Minimum Value Point to End Point Time"],
                        statisticsFromDeviationSequences["maximum_value_portion"],
                        statisticsFromDeviationSequences["minimum_value_portion"],
                        statisticsFromDeviationVelocitySequences["Kurtosis"],
                        statisticsFromDeviationVelocitySequences["Mean"],
                        statisticsFromDeviationVelocitySequences["Mean Before Max Deviation Point"],
                        statisticsFromDeviationVelocitySequences["Mean After Max Deviation Point"],
                        statisticsFromDeviationVelocitySequences["Minimum"],
                        statisticsFromDeviationVelocitySequences["Maximum"],
                        statisticsFromDeviationVelocitySequences["Standard Deviation"],
                        statisticsFromDeviationVelocitySequences["Median"],
                        statisticsFromDeviationVelocitySequences["Lower Quartile"],
                        statisticsFromDeviationVelocitySequences["Second Quartile"],
                        statisticsFromDeviationVelocitySequences["Third Quartile"],
                        statisticsFromDeviationVelocitySequences["Quadratic Mean"],
                        statisticsFromDeviationVelocitySequences["Harmonic Mean"],
                        statisticsFromDeviationVelocitySequences["Geometric Mean"],
                        statisticsFromDeviationVelocitySequences["Mean Absolute Deviation"],
                        statisticsFromDeviationVelocitySequences["First Point value"],
                        statisticsFromDeviationVelocitySequences["Last Point value"],
                        statisticsFromDeviationVelocitySequences["Twenty Percent"],
                        statisticsFromDeviationVelocitySequences["Eighty Percent"],
                        statisticsFromDeviationVelocitySequences["Largest Deviation Point Value"],
                        statisticsFromDeviationVelocitySequences["Time to reach Maximum Value"],
                        statisticsFromDeviationVelocitySequences["Time to reach Minimum Value"],
                        statisticsFromDeviationVelocitySequences["Length to reach Maximum Value"],
                        statisticsFromDeviationVelocitySequences["Length to reach Minimum Value"],
                        statisticsFromDeviationVelocitySequences["Length of Maximum Value to End"],
                        statisticsFromDeviationVelocitySequences["Length of Minimum Value to End"],
                        statisticsFromDeviationVelocitySequences["Skewness"],
                        statisticsFromDeviationVelocitySequences["Variance"],
                        statisticsFromDeviationVelocitySequences["Coefcient of Variation"],
                        statisticsFromDeviationVelocitySequences["Standard Error of the Mean"],
                        statisticsFromDeviationVelocitySequences["Interquartile Range"],
                        statisticsFromDeviationVelocitySequences["Median Absolute Deviation"],
                        statisticsFromDeviationVelocitySequences["Median Value at First 3 Points"],
                        statisticsFromDeviationVelocitySequences["Median Value at First 5 Points"],
                        statisticsFromDeviationVelocitySequences["Median Value at Last 3 Points"],
                        statisticsFromDeviationVelocitySequences["Median Value at Last 5 Points"],
                        statisticsFromDeviationVelocitySequences["Maximum Value Point to End Point Time"],
                        statisticsFromDeviationVelocitySequences["Minimum Value Point to End Point Time"],
                        statisticsFromDeviationVelocitySequences["maximum_value_portion"],
                        statisticsFromDeviationVelocitySequences["minimum_value_portion"],
                        statisticsFromDeviationAccSequences["Kurtosis"],
                        statisticsFromDeviationAccSequences["Mean"],
                        statisticsFromDeviationAccSequences["Mean Before Max Deviation Point"],
                        statisticsFromDeviationAccSequences["Mean After Max Deviation Point"],
                        statisticsFromDeviationAccSequences["Minimum"],
                        statisticsFromDeviationAccSequences["Maximum"],
                        statisticsFromDeviationAccSequences["Standard Deviation"],
                        statisticsFromDeviationAccSequences["Median"],
                        statisticsFromDeviationAccSequences["Lower Quartile"],
                        statisticsFromDeviationAccSequences["Second Quartile"],
                        statisticsFromDeviationAccSequences["Third Quartile"],
                        statisticsFromDeviationAccSequences["Quadratic Mean"],
                        statisticsFromDeviationAccSequences["Harmonic Mean"],
                        statisticsFromDeviationAccSequences["Geometric Mean"],
                        statisticsFromDeviationAccSequences["Mean Absolute Deviation"],
                        statisticsFromDeviationAccSequences["First Point value"],
                        statisticsFromDeviationAccSequences["Last Point value"],
                        statisticsFromDeviationAccSequences["Twenty Percent"],
                        statisticsFromDeviationAccSequences["Eighty Percent"],
                        statisticsFromDeviationAccSequences["Largest Deviation Point Value"],
                        statisticsFromDeviationAccSequences["Time to reach Maximum Value"],
                        statisticsFromDeviationAccSequences["Time to reach Minimum Value"],
                        statisticsFromDeviationAccSequences["Length to reach Maximum Value"],
                        statisticsFromDeviationAccSequences["Length to reach Minimum Value"],
                        statisticsFromDeviationAccSequences["Length of Maximum Value to End"],
                        statisticsFromDeviationAccSequences["Length of Minimum Value to End"],
                        statisticsFromDeviationAccSequences["Skewness"],
                        statisticsFromDeviationAccSequences["Variance"],
                        statisticsFromDeviationAccSequences["Coefcient of Variation"],
                        statisticsFromDeviationAccSequences["Standard Error of the Mean"],
                        statisticsFromDeviationAccSequences["Interquartile Range"],
                        statisticsFromDeviationAccSequences["Median Absolute Deviation"],
                        statisticsFromDeviationAccSequences["Median Value at First 3 Points"],
                        statisticsFromDeviationAccSequences["Median Value at First 5 Points"],
                        statisticsFromDeviationAccSequences["Median Value at Last 3 Points"],
                        statisticsFromDeviationAccSequences["Median Value at Last 5 Points"],
                        statisticsFromDeviationAccSequences["Maximum Value Point to End Point Time"],
                        statisticsFromDeviationAccSequences["Minimum Value Point to End Point Time"],
                        statisticsFromDeviationAccSequences["maximum_value_portion"],
                        statisticsFromDeviationAccSequences["minimum_value_portion"],
                        dir,
                    ]
                    swipeFeatures.insert(0, discard_swipe_counter)
                    swipeFeatures.insert(0, swipe_counter)
                    swipeFeatures.insert(0, path.split("/")[-3])
                    swipeFeatures.insert(0, path.split("/")[-4])
                    swipeFeatures.insert(0, path.split("/")[-5])
                    swipeFeatures.insert(0, model)
                    swipeFeatures.insert(0, path.split("/")[-6])
                    output.append(swipeFeatures)
                    swipe_counter += 1
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            f = open(export_path, "w+")
            for o in output:
                f.write(",".join(str(e) for e in o) + "\n")
            f.close()
    features_folder = "data_files"
    export_path = "features.csv"
    if os.path.exists(export_path):
        os.remove(export_path)
    f_export = open(export_path, "a+")
    userdata = pd.read_csv("tables/userdata.csv")
    cuuid = ""
    total = len(userdata.index)
    current = 0
    t_first = True
    for subdir, dirs, files in os.walk(features_folder):
        for file in files:
            path = os.path.join(subdir, file)
            if path.split("/")[-1] != "features.csv":
                continue
            uuid = path.split("/")[-5]
            if uuid != cuuid:
                cuuid = uuid
                current += 1
                sys.stdout.write("\rUser (%d/%d)" % (current, total))
                sys.stdout.flush()
            first = True
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if first:
                        first = False
                        if t_first:
                            f_export.write(line)
                            t_first = False
                    else:
                        f_export.write(line)
    f_export.close()
