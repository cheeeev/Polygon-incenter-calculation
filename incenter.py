import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, LineString, Point
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
from collections import deque


# 数据读取函数
def read_data(file_name):
    points = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                points.append([x, y])
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    return np.array(points)


def read_shp_data(file_name):
    try:
        # 读取shp文件
        gdf = gpd.read_file(file_name)

        # 创建一个用于从 WGS84 (经纬度, EPSG:4326) 转换到 Web Mercator (以米为单位, EPSG:3857) 的转换器
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # 假设每个Polygon对象是一个MultiPolygon或Polygon
        polygons = gdf.geometry
        print(polygons)

        # 提取所有多边形的坐标点，并进行坐标转换
        points = []
        for poly in polygons:
            if poly.is_valid:
                # 检查是否是MultiPolygon
                if poly.geom_type == 'MultiPolygon':
                    for sub_poly in poly:
                        # 将每个多边形的经纬度坐标转换为以米为单位的坐标
                        transformed_coords = [transformer.transform(x, y) for x, y in sub_poly.exterior.coords]
                        points.extend(transformed_coords)
                else:
                    # 将多边形的经纬度坐标转换为以米为单位的坐标
                    transformed_coords = [transformer.transform(x, y) for x, y in poly.exterior.coords]
                    points.extend(transformed_coords)

        # 将点转换为numpy数组
        return np.array(points)

    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None

# 计算凸包面积
def convex_hull_area(points):
    if points is None or len(points) < 3:
        print("Insufficient points to form a convex hull.")
        return None
    hull = ConvexHull(points)
    return hull.volume
# 计算多边形面积
def polygon_area(points):
    shapely_polygon = Polygon(points)
    return shapely_polygon.area, shapely_polygon.centroid


def calculate_delta(tri_coords, polygon):
    polygonarea = polygon.area
    # print(polygonarea)

    # 创建 Polygon 对象
    triangle = Polygon(tri_coords)

    # 确保三角形在多边形内，否则将不进行计算
    if not polygon.contains(triangle):
        print("三角形不在多边形内，无法计算。")
        return 1

    # 使用 difference() 函数将多边形减去三角形，得到剩余部分
    remaining_area = polygon.difference(triangle)

    # 处理多个部分，如果剩余的区域是多个部分
    if remaining_area.geom_type == 'MultiPolygon':
        areas = [part.area for part in remaining_area.geoms]
    else:
        areas = [remaining_area.area]

    target_area = polygonarea / 3
    # 输出每个部分的面积
    # for i, area in enumerate(areas):
    #     print(f"Part {i + 1} area: {area}")
    delta = 0.5 * sum((area - target_area) ** 2 for area in areas)
    return delta


def get_triangles(points):
    triangles = []
    tri = Delaunay(points)
    for simplex in tri.simplices:  # 遍历每个三角形
        triangle_points = points[simplex]  # 获取三角形的三个点
        triangle = Polygon(triangle_points)  # 创建 Shapely Polygon
        triangles.append(triangle)  # 添加到列表中
    return triangles


def filter_triangles(triangles, polygon):
    filtered_triangles = []
    for triangle in triangles:
        intersection_area = triangle.intersection(polygon).area  # 计算相交面积
        triangle_area = triangle.area
        if intersection_area == triangle_area:  # 仅保留相交面积大于0的三角形
            filtered_triangles.append(triangle)
    return filtered_triangles


def get_boundary_lines(polygon):
    # 提取多边形的边界线
    boundary = polygon.exterior
    lines = []
    for i in range(len(boundary.coords) - 1):
        line = LineString([boundary.coords[i], boundary.coords[i + 1]])  # 创建线段
        lines.append(line)  # 添加到线段列表中
    return lines


def get_triangle_boundaries(triangles):
    triangle_boundaries = []
    for triangle in triangles:
        boundary = triangle.exterior  # 获取三角形的边界
        lines = []
        for i in range(len(boundary.coords) - 1):
            line = LineString([boundary.coords[i], boundary.coords[i + 1]])  # 创建线段
            lines.append(line)  # 添加到线段列表中
        triangle_boundaries.append(lines)  # 添加该三角形的边界线段
    return triangle_boundaries

def triangle_delta(triangle_boundaries, polygon_lines, polygon):
    def extract_unique_points(boundaries):
        # 创建一个集合来存储唯一的端点
        unique_points = set()

        # 遍历每个 LINESTRING 类型的边界线段
        for boundary in boundaries:
            # boundary.coords 返回线段的所有点
            for point in boundary.coords:
                unique_points.add(point)  # 使用集合自动去除重复点

        # 将集合转换为列表并返回
        return list(unique_points)

    # 创建一个字典来存储 delta 和 tri_centroid 的对应关系
    delta_centroid_dict = {}

    for boundaries in triangle_boundaries:
        count = 0

        for boundary in boundaries:
            for line in polygon_lines:
                # 检查两个线段是否相等
                if boundary.equals(line):  # 只有在两个端点都相同的情况下才计数
                    count += 1

        # 如果 count 为 0，处理对应的 boundaries
        if count == 0:
            # 提取三角形的端点
            tri_boundary = extract_unique_points(boundaries)

            # 计算 delta 值
            delta = calculate_delta(tri_boundary, polygon)
            tri_polygon = Polygon(tri_boundary)
            # 计算三角形的质心
            tri_centroid = tri_polygon.centroid

            # 将 delta 和 tri_centroid 存入字典
            delta_centroid_dict[delta] = tri_centroid
            # print(delta_centroid_dict)

    # 返回字典
    return delta_centroid_dict








def count_intersections(triangle_boundaries, polygon_lines):
    intersections_count = []  # 存储每个三角形的相交线段数
    common_vertices = []  # 存储找到的公共顶点

    for boundaries in triangle_boundaries:
        count = 0
        intersecting_boundaries = []  # 用于存储相交的 boundary
        for boundary in boundaries:
            for line in polygon_lines:
                # 检查两个线段是否相等
                if boundary.equals(line):  # 只有在两个端点都相同的情况下才计数
                    count += 1
                    intersecting_boundaries.append(boundary)  # 保存相等的 boundary

        intersections_count.append(count)  # 添加每个三角形的相交线段数

        # 如果找到2条相等的边界，检查它们是否有公共顶点
        if count == 2 and len(intersecting_boundaries) == 2:
            # 获取第一个 boundary 的两个端点
            boundary1_coords = list(intersecting_boundaries[0].coords)
            # 获取第二个 boundary 的两个端点
            boundary2_coords = list(intersecting_boundaries[1].coords)

            # 找到两个 boundary 的公共顶点
            common_point = set(boundary1_coords).intersection(boundary2_coords)

            if common_point:  # 如果找到了公共顶点
                common_vertices.append(list(common_point)[0])  # 添加最末端顶点

    return intersections_count, common_vertices  # 返回相交数目和最末端顶点


def class_3_center_midpoints(midpoints_list):
    if len(midpoints_list) != 3:
        raise ValueError("输入的坐标列表必须包含三个坐标")
    sum_x = sum(coord[0] for coord in midpoints_list)
    sum_y = sum(coord[1] for coord in midpoints_list)
    mean_x = round(sum_x / 3, 2)
    mean_y = round(sum_y / 3, 2)
    return (mean_x, mean_y)


def insert_vertices(skeleton, common_vertices):
    # 遍历 skeleton 中的每条折线
    for line in skeleton:
        if line:  # 确保该折线非空
            first_point = line[0]  # 获取折线的第一个坐标

            # 存储该折线第一个点与 common_vertices 中每个点的距离
            distances_for_first_point = []
            for vertex in common_vertices:
                # 计算欧几里得距离
                distance = np.sqrt((first_point[0] - vertex[0]) ** 2 + (first_point[1] - vertex[1]) ** 2)
                distances_for_first_point.append(distance)

            # 找到最小距离和对应的索引
            min_distance = min(distances_for_first_point)
            min_index = distances_for_first_point.index(min_distance)

            # 获取最小距离对应的 common_vertices 值
            closest_vertex = common_vertices[min_index]

            # 将这个点插入到 line 的第一个元素之前
            line.insert(0, closest_vertex)

    return skeleton

def get_middlepoints(triangle_boundaries, polygon_lines):
    middlepoints = []  # 存储所有三角形的中点列表
    for boundaries in triangle_boundaries:
        boundary_midpoints = []  # 存储当前三角形的所有中点
        for boundary in boundaries:
            # 如果 polygon_lines 中有任何一个 line 和 boundary 相等，跳过该 boundary
            if any(boundary.equals(line) for line in polygon_lines):
                continue  # 跳过相等的线段

            # 计算线段的中点
            midpoint = boundary.interpolate(0.5, normalized=True)  # 计算中点
            midpoint_coords = midpoint.coords[0]  # 获取中点坐标

            # 检查中点是否已存在于 boundary_midpoints 列表中
            if tuple(midpoint_coords) not in boundary_midpoints:
                boundary_midpoints.append(tuple(midpoint_coords))  # 将中点坐标添加到列表中

        # 如果当前三角形的中点列表非空，则将其添加到 middlepoints
        if boundary_midpoints:
            middlepoints.append(boundary_midpoints)

    return middlepoints

def classify_triangles(intersections_count):
    # 分类三角形
    class_1 = []  # intersections_count = 2
    class_2 = []  # intersections_count = 1
    class_3 = []  # intersections_count = 0

    for count in intersections_count:
        if count == 2:
            class_1.append(count)
        elif count == 1:
            class_2.append(count)
        elif count == 0:
            class_3.append(count)

    return class_1, class_2, class_3

def GenSkeleton(middlepoints):
    print("start creating corner skeleton")
    skeleton = []  # 存储多个中点的列表

    # 主循环：不断从 middlepoints 中查找仅包含一个中点的列表，直到 middlepoints 中所有列表长度不为 1
    while any(len(midpoints_list) == 1 for midpoints_list in middlepoints):  # 判断是否还有长度为 1 的列表
        separate_skeleton = []  # 存储仅有一个中点的列表
        found_midpoint = None  # 用于记录找到的单个中点

        # 遍历 middlepoints，查找仅包含一个中点的列表
        for midpoints_list in middlepoints:
            if len(midpoints_list) == 1:  # 如果列表中只有一个中点
                separate_skeleton.append(midpoints_list[0])  # 将中点坐标添加到 separate_skeleton
                found_midpoint = midpoints_list[0]  # 记录找到的单个中点
                middlepoints.remove(midpoints_list)  # 从 middlepoints 中移除此列表
                break  # 停止当前循环，开始查找与 found_midpoint 相连的列表

        # 如果找到了一个中点，继续查找与 found_midpoint 相同的坐标
        while found_midpoint:
            found_next = False  # 标记是否找到下一个点
            for midpoints_list in middlepoints[:]:  # 使用 [:] 来遍历时不影响原列表
                if found_midpoint in midpoints_list:
                    if len(midpoints_list) == 2:  # 如果该列表长度为 2
                        # 找到与 found_midpoint 不同的那个点
                        other_point = midpoints_list[0] if midpoints_list[1] == found_midpoint else midpoints_list[1]
                        separate_skeleton.append(other_point)  # 将其他点添加到 separate_skeleton
                        found_midpoint = other_point  # 将 other_point 作为新的 found_midpoint
                        middlepoints.remove(midpoints_list)  # 删除这个列表
                        found_next = True  # 标记为找到
                        break  # 重新开始查找
                    elif len(midpoints_list) == 3:  # 如果列表长度为 3
                        found_midpoint = None  # 停止查找
                        class_3_centroid = class_3_center_midpoints(midpoints_list)  # 获取重心
                        separate_skeleton.append(class_3_centroid)  # 添加重心到separate_skeleton
                        break  # 退出循环

            if not found_next:  # 如果没有找到下一个点，则停止循环
                break

        # 如果停止查找，将 separate_skeleton 添加到 skeleton 并清空它
        if separate_skeleton:
            skeleton.append(separate_skeleton.copy())  # 添加副本，保留原内容
            separate_skeleton.clear()  # 清空 separate_skeleton 列表
        print(len(middlepoints))
        # plot_skeleton(skeleton)
        # plot_triangles(filtered_triangles, intersections_count)
        # plt.savefig("test.png")


    return skeleton, middlepoints


def get_middle_skeleton(new_middlepoints, skeleton, filtered_triangles, intersections_count):
    print("start creating middle skeleton")

    # 用 deque 存储最近 5 次的 new_middlepoints 长度
    recent_lengths = deque(maxlen=5)

    while len(new_middlepoints) > 1:
        # 记录当前 new_middlepoints 的长度
        recent_lengths.append(len(new_middlepoints))

        # 如果最近 5 次长度都相同，直接返回 skeleton
        if len(recent_lengths) == 5 and len(set(recent_lengths)) == 1:
            print("The length of new_middlepoints has been the same for 5 consecutive loops, exiting early.")
            return new_middlepoints, skeleton

        for idx, midpoints_list in enumerate(new_middlepoints):
            if len(midpoints_list) == 3:
                matching_points = []
                middle_skeleton = []

                for skeleton_list in skeleton:
                    for point in midpoints_list:
                        if point in skeleton_list:
                            matching_points.append(point)

                if len(matching_points) == 2:
                    centroid = class_3_center_midpoints(midpoints_list)
                    midpoints_list = [point for point in midpoints_list if point not in matching_points]

                    new_middlepoints[idx] = midpoints_list
                    break

        skeleton_middle, new_middlepoints = GenSkeleton(new_middlepoints)
        skeleton_middle = [item for sublist in skeleton_middle for item in sublist]
        skeleton_middle.insert(0, centroid)
        skeleton.append(skeleton_middle)
        print(len(new_middlepoints))

    return new_middlepoints, skeleton


def find_split_point(coordinate_lists):
    print("start creating graph")
    # 创建一个无向图
    G = nx.Graph()

    # 添加节点和边
    for coord_list in coordinate_lists:
        for i in range(len(coord_list)):
            G.add_node(coord_list[i])  # 添加节点
            # 连接相邻的节点
            if i > 0:
                G.add_edge(coord_list[i - 1], coord_list[i])

    # 找出最远的两个节点及其路径
    longest_distance = 0
    longest_path = None
    longest_nodes = (None, None)

    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                try:
                    length = nx.shortest_path_length(G, source=node1, target=node2)
                    if length > longest_distance:
                        longest_distance = length
                        longest_path = nx.shortest_path(G, source=node1, target=node2)
                        longest_nodes = (node1, node2)
                except nx.NetworkXNoPath:
                    continue  # 如果没有路径则跳过

    # 将最远路径转化为polyline
    polyline = [(x, y) for (x, y) in longest_path]

    # 计算polyline的总长度
    def calculate_length(points):
        total_length = 0.0
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(np.array(points[i]) - np.array(points[i + 1]))
        return total_length

    total_length = calculate_length(polyline)

    # 查找均分点
    half_length = total_length / 2.0
    current_length = 0.0
    split_point = None

    for i in range(len(polyline) - 1):
        segment_length = np.linalg.norm(np.array(polyline[i]) - np.array(polyline[i + 1]))
        if current_length + segment_length >= half_length:
            # 找到均分点
            excess_length = half_length - current_length
            direction = np.array(polyline[i + 1]) - np.array(polyline[i])
            direction = direction / np.linalg.norm(direction)  # 单位向量
            split_point = np.array(polyline[i]) + direction * excess_length
            break
        current_length += segment_length

    return split_point.tolist()

def plot_triangles(filtered_triangles, intersections_count):
    print("start drawing triangles")
    # 为每一类三角形指定不同的颜色
    colors = {1: 'green', 2: 'blue', 3: 'red'}

    for i, triangle in enumerate(filtered_triangles):
        count = intersections_count[i]
        color = colors.get(count, 'black')  # 默认颜色为黑色
        x, y = triangle.exterior.xy
        plt.fill(x, y, alpha=0.2, fc=color, ec='black')  # 绘制三角形
def plot_skeleton(skeleton):
    print("start drawing skeleton")
    # 遍历每条折线
    for line in skeleton:
        # 将每条线的点分开为 x 和 y 坐标
        x_values = [point[0] for point in line]
        y_values = [point[1] for point in line]

        # plt.plot(x_values, y_values, marker='o', color='blue')  # 绘制折线，带上点标记
def plot_polygon(polygon):
    print("start drawing polygon")
    x, y = polygon.exterior.xy  # 获取多边形外部的x和y坐标
    plt.fill(x, y, alpha=0.5, fc='lightblue', ec='black')  # 绘制多边形

def plot_graph(coordinate_lists, split_point):
    print("start drawing graph")
    # 创建无向图
    G = nx.Graph()

    # 添加节点和边
    for coord_list in coordinate_lists:
        for i in range(len(coord_list)):
            G.add_node(coord_list[i])  # 添加节点
            # 连接相邻的节点
            if i > 0:
                G.add_edge(coord_list[i - 1], coord_list[i])

    # 找出最远的两个节点及其路径
    longest_distance = 0
    longest_path = None
    longest_nodes = (None, None)

    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                try:
                    length = nx.shortest_path_length(G, source=node1, target=node2)
                    if length > longest_distance:
                        longest_distance = length
                        longest_path = nx.shortest_path(G, source=node1, target=node2)
                        longest_nodes = (node1, node2)
                except nx.NetworkXNoPath:
                    continue  # 如果没有路径则跳过

    # 将最远路径转化为polyline
    polyline = [(x, y) for (x, y) in longest_path]

    # 绘制图形
    pos = {coord: coord for coord in G.nodes()}  # 节点位置
    nx.draw(G, pos, with_labels=False, node_size=3, node_color='lightblue', font_size=1)

    # 突出显示最远节点的路径
    if longest_path:
        path_edges = list(zip(longest_path[:-1], longest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    # 标记均分点
    if split_point is not None:
        plt.plot(split_point[0], split_point[1], 'ro', markersize=10, label="内心")  # 绿色圆点
        # plt.text(split_point[0], split_point[1]+20, 'Incenter', fontsize=15, ha='center')
        plt.legend()

def plot_centroid(polygon):
    centroid = polygon.centroid
    xmin, ymin, xmax, ymax = polygon.bounds
    plt.plot(centroid.x, centroid.y, marker='o', markersize=10, color="green", label="重心")
    # plt.text(centroid.x, centroid.y+20, 'Centroid', fontsize=15, ha='center')
    plt.plot((xmin + xmax)/2, (ymin + ymax)/2, marker='*', markersize=10, color="green", label="MBR中心")
    # plt.text((xmin + xmax)/2, (ymin + ymax)/2+20, 'C$_E$', fontsize=15, ha='center', wrap=True)
    plt.legend()




# 主函数
def main(points):
    file_name = 'path/to/your/file.shp'  # 读取合并后的shp文件
    points = read_shp_data(file_name)

    if points is not None:
        print(f"Read {len(points)} points.")
        triangles = get_triangles(points)  # 获取所有三角形

        polygon = Polygon(points)  # 使用 Shapely 创建多边形
        polygon_area_value, centroid = polygon_area(points)  # 计算多边形的面积和重心
        convexhullarea = convex_hull_area(points)

        # 计算面积比
        area_ratio = polygon_area_value / convexhullarea
        print(f"Area ratio: {area_ratio}")

        if area_ratio > 0.85:
            print(f"内心坐标: {centroid.x}, {centroid.y}")
            return centroid
        else:
            # 过滤相交面积为0的三角形
            filtered_triangles = filter_triangles(triangles, polygon)

            # 提取多边形的边界线
            polygon_lines = get_boundary_lines(polygon)

            # 提取每个过滤三角形的边界
            triangle_boundaries = get_triangle_boundaries(filtered_triangles)

            # 计算每个三角形的边界与多边形的边界相交的线段数
            intersections_count, common_vertices = count_intersections(triangle_boundaries, polygon_lines)
            delta_centroid_dict = triangle_delta(triangle_boundaries, polygon_lines, polygon)
            # print(delta_centroid_dict)
            # 分类三角形
            class_1, class_2, class_3 = classify_triangles(intersections_count)

            # 输出每一类三角形的个数
            print(f"Class 1 triangles (intersections_count = 2): {len(class_1)}")
            print(f"Class 2 triangles (intersections_count = 1): {len(class_2)}")
            print(f"Class 3 triangles (intersections_count = 0): {len(class_3)}")

            min_delta = min(delta_centroid_dict.keys())
            print(f"最小的delta为{min_delta}")
            tri_centroid = delta_centroid_dict[min_delta]  # 对应的三角形质心



            # 判断最小的 delta 是否大于 0.65
            if min_delta > 0.65:
                # 获取中点坐标
                middlepoints = get_middlepoints(triangle_boundaries, polygon_lines)
                middlepoints = [[(round(x, 2), round(y, 2)) for x, y in sublist] for sublist in middlepoints]

                # 获取顶点三角形到节点三角形的骨架
                skeleton, new_middlepoints = GenSkeleton(middlepoints)
                skeleton = insert_vertices(skeleton, common_vertices)

                # 获取节点三角形到节点三角形的骨架
                new_middlepoints, skeleton = get_middle_skeleton(new_middlepoints, skeleton, filtered_triangles, intersections_count)
                plot_triangles(filtered_triangles, intersections_count)
                plot_skeleton(skeleton)
                plt.show()
                split_point = find_split_point(skeleton)
                print("内心坐标:", split_point)

                # 绘制图形
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                plot_polygon(polygon)
                plot_skeleton(skeleton)
                plot_centroid(polygon)
                plot_triangles(filtered_triangles, intersections_count)
                plot_graph(skeleton, split_point)
                plt.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='red')  # 绘制点
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid()
                plt.show()

                point = Point(split_point)
                return point

            else:
                # 输出三角形质心
                print(f"内心坐标为 {tri_centroid}")

                # 绘制图形
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                plot_polygon(polygon)
                plot_centroid(polygon)
                plot_triangles(filtered_triangles, intersections_count)
                plt.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='red')  # 绘制点
                plt.plot(tri_centroid.x, tri_centroid.y, 'ro', markersize=10, label='Incenter')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.legend()
                plt.grid()
                plt.show()
                return tri_centroid
    else:
        print("No points to process.")




incenter = main(None)
print(incenter)
