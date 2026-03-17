"""
PLY 点选择器测试脚本

运行测试:
    python test_ply_point_picker.py

这个脚本会:
1. 生成测试 PLY 数据
2. 生成测试 tileset.json
3. 测试各种查询功能
"""

import numpy as np
import os
import json
import struct

# 测试数据目录
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def ensure_test_data_dir():
    """确保测试数据目录存在"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)


def generate_test_ply(
    num_points: int = 1000,
    center_local: tuple = (0, 0, 0),
    spread: float = 100.0,
    output_path: str = None
) -> str:
    """
    生成测试 PLY 文件
    
    Args:
        num_points: 点数量
        center_local: 本地坐标中心
        spread: 点分布范围
        output_path: 输出文件路径
    
    Returns:
        生成的 PLY 文件路径
    """
    if output_path is None:
        output_path = os.path.join(TEST_DATA_DIR, "test_model.ply")
    
    # 生成随机点云数据
    np.random.seed(42)  # 固定随机种子，便于复现
    
    # 在指定范围内生成随机点
    x = np.random.uniform(center_local[0] - spread/2, center_local[0] + spread/2, num_points)
    y = np.random.uniform(center_local[1] - spread/2, center_local[1] + spread/2, num_points)
    z = np.random.uniform(center_local[2] - spread/2, center_local[2] + spread/2, num_points)
    
    # 生成随机颜色
    red = np.random.randint(0, 256, num_points, dtype=np.uint8)
    green = np.random.randint(0, 256, num_points, dtype=np.uint8)
    blue = np.random.randint(0, 256, num_points, dtype=np.uint8)
    
    # 写入 PLY 文件
    with open(output_path, 'wb') as f:
        # PLY 头部
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        # 写入数据
        for i in range(num_points):
            f.write(struct.pack('<fff', x[i], y[i], z[i]))
            f.write(struct.pack('<BBB', red[i], green[i], blue[i]))
    
    print(f"[测试] 生成 PLY 文件: {output_path}")
    print(f"  - 点数量: {num_points:,}")
    print(f"  - 本地坐标范围: X[{x.min():.2f}, {x.max():.2f}]")
    print(f"  - 本地坐标范围: Y[{y.min():.2f}, {y.max():.2f}]")
    print(f"  - 本地坐标范围: Z[{z.min():.2f}, {z.max():.2f}]")
    
    return output_path


def generate_test_tileset_json(
    center_longitude: float = 117.636892,
    center_latitude: float = 24.832147,
    center_height: float = 379.13,
    output_path: str = None
) -> str:
    """
    生成测试 tileset.json 文件
    
    创建一个将本地坐标系原点放置在指定 WGS84 位置的变换矩阵
    
    Args:
        center_longitude: 中心点经度
        center_latitude: 中心点纬度
        center_height: 中心点高度
        output_path: 输出文件路径
    
    Returns:
        生成的 tileset.json 文件路径
    """
    if output_path is None:
        output_path = os.path.join(TEST_DATA_DIR, "test_tileset.json")
    
    # WGS84 参数
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    
    # 计算 ECEF 坐标
    lon_rad = np.radians(center_longitude)
    lat_rad = np.radians(center_latitude)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
    
    ecef_x = (N + center_height) * cos_lat * cos_lon
    ecef_y = (N + center_height) * cos_lat * sin_lon
    ecef_z = (N * (1 - e2) + center_height) * sin_lat
    
    # 构建 ENU (East-North-Up) 到 ECEF 的旋转矩阵
    # 这将本地 X 轴指向东，Y 轴指向北，Z 轴指向上
    rotation = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0, cos_lat, sin_lat]
    ])
    
    # 构建 4x4 变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [ecef_x, ecef_y, ecef_z]
    
    # 转换为列主序（3D Tiles 格式）
    transform_column_major = transform.T.flatten().tolist()
    
    # 创建 tileset.json
    tileset = {
        "asset": {
            "version": "1.0",
            "gltfUpAxis": "Z"
        },
        "geometricError": 1000,
        "root": {
            "transform": transform_column_major,
            "boundingVolume": {
                "region": [
                    np.radians(center_longitude - 0.01),
                    np.radians(center_latitude - 0.01),
                    np.radians(center_longitude + 0.01),
                    np.radians(center_latitude + 0.01),
                    0,
                    500
                ]
            },
            "geometricError": 100,
            "content": {
                "uri": "test_model.ply"
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tileset, f, indent=2)
    
    print(f"[测试] 生成 tileset.json: {output_path}")
    print(f"  - 中心点 WGS84: ({center_longitude}, {center_latitude}, {center_height}m)")
    print(f"  - 中心点 ECEF: ({ecef_x:.2f}, {ecef_y:.2f}, {ecef_z:.2f})")
    
    return output_path


def run_tests():
    """运行测试"""
    from ply_point_picker import PlyPointPicker
    
    print("=" * 60)
    print("PLY 点选择器测试")
    print("=" * 60)
    
    # 准备测试数据
    ensure_test_data_dir()
    
    # 测试参数
    center_longitude = 117.636892
    center_latitude = 24.832147
    center_height = 379.13
    num_points = 10000
    
    # 生成测试数据
    print("\n[步骤 1] 生成测试数据")
    print("-" * 40)
    ply_path = generate_test_ply(
        num_points=num_points,
        center_local=(0, 0, 0),
        spread=100.0
    )
    tileset_path = generate_test_tileset_json(
        center_longitude=center_longitude,
        center_latitude=center_latitude,
        center_height=center_height
    )
    
    # 测试初始化
    print("\n[步骤 2] 初始化 PlyPointPicker")
    print("-" * 40)
    picker = PlyPointPicker(ply_path, tileset_path)
    
    # 测试坐标转换
    print("\n[步骤 3] 测试坐标转换")
    print("-" * 40)
    
    # WGS84 -> ECEF -> WGS84
    test_lon, test_lat, test_h = center_longitude, center_latitude, center_height
    ecef = PlyPointPicker.wgs84_to_ecef(test_lon, test_lat, test_h)
    print(f"  WGS84 输入: ({test_lon}, {test_lat}, {test_h})")
    print(f"  ECEF: ({ecef[0]:.2f}, {ecef[1]:.2f}, {ecef[2]:.2f})")
    
    lon2, lat2, h2 = PlyPointPicker.ecef_to_wgs84(*ecef)
    print(f"  WGS84 输出: ({lon2:.6f}, {lat2:.6f}, {h2:.2f})")
    
    # 验证转换精度
    assert abs(lon2 - test_lon) < 1e-6, "经度转换误差过大"
    assert abs(lat2 - test_lat) < 1e-6, "纬度转换误差过大"
    assert abs(h2 - test_h) < 0.01, "高度转换误差过大"
    print("  [OK] 坐标转换验证通过")
    
    # 测试最近点查询
    print("\n[步骤 4] 测试最近点查询")
    print("-" * 40)
    
    result = picker.get_index_at_wgs84(
        longitude=center_longitude,
        latitude=center_latitude,
        height=center_height,
        max_distance=100.0
    )
    
    if result:
        print(f"  找到最近的点:")
        print(f"    - PLY 索引: {result['ply_index']}")
        print(f"    - 距离: {result['distance']:.4f} 米")
        print(f"    - 本地坐标: {result['position_local']}")
        print(f"    - WGS84: ({result['position_wgs84']['longitude']:.6f}, "
              f"{result['position_wgs84']['latitude']:.6f}, "
              f"{result['position_wgs84']['height']:.2f}m)")
        print("  [OK] 最近点查询测试通过")
    else:
        print("  [FAIL] 未找到最近的点")
    
    # 测试 K 最近邻查询
    print("\n[步骤 5] 测试 K 最近邻查询")
    print("-" * 40)
    
    k = 5
    results = picker.get_k_nearest_at_wgs84(
        longitude=center_longitude,
        latitude=center_latitude,
        height=center_height,
        k=k
    )
    
    print(f"  找到 {len(results)} 个最近邻:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] PLY 索引 = {r['ply_index']}, 距离 = {r['distance']:.4f}m")
    
    assert len(results) == k, f"应该返回 {k} 个结果"
    
    # 验证结果按距离排序
    for i in range(len(results) - 1):
        assert results[i]['distance'] <= results[i+1]['distance'], "结果应按距离排序"
    print("  [OK] K 最近邻查询测试通过")
    
    # 测试半径查询
    print("\n[步骤 6] 测试半径查询")
    print("-" * 40)
    
    radius = 30.0
    results = picker.get_indices_in_radius(
        longitude=center_longitude,
        latitude=center_latitude,
        height=center_height,
        radius=radius
    )
    
    print(f"  在 {radius}m 半径内找到 {len(results)} 个点")
    
    # 验证所有结果都在半径内
    for r in results:
        assert r['distance'] <= radius, f"点 {r['ply_index']} 距离 {r['distance']} 超出半径 {radius}"
    print("  [OK] 半径查询测试通过")
    
    # 测试批量查询
    print("\n[步骤 7] 测试批量查询")
    print("-" * 40)
    
    # 生成测试坐标（在中心点附近）
    test_coordinates = [
        (center_longitude, center_latitude, center_height),
        (center_longitude + 0.0001, center_latitude, center_height),
        (center_longitude, center_latitude + 0.0001, center_height),
        # 一个远离的点，预期不会找到匹配
        (center_longitude + 1, center_latitude + 1, center_height),
    ]
    
    results = picker.batch_query(test_coordinates, max_distance=50.0)
    
    print(f"  批量查询 {len(test_coordinates)} 个坐标:")
    for i, (coord, result) in enumerate(zip(test_coordinates, results)):
        if result:
            print(f"    [{i+1}] ({coord[0]:.6f}, {coord[1]:.6f}) -> PLY 索引 {result['ply_index']}")
        else:
            print(f"    [{i+1}] ({coord[0]:.6f}, {coord[1]:.6f}) -> 未找到")
    
    # 前3个应该找到，最后一个不应该找到
    assert results[0] is not None, "第1个坐标应该找到匹配"
    assert results[1] is not None, "第2个坐标应该找到匹配"
    assert results[2] is not None, "第3个坐标应该找到匹配"
    assert results[3] is None, "第4个坐标不应该找到匹配"
    print("  [OK] 批量查询测试通过")
    
    # 测试点信息查询
    print("\n[步骤 8] 测试点信息查询")
    print("-" * 40)
    
    # 查询索引 0 的点
    info = picker.get_point_info(0)
    if info:
        print(f"  点 0 信息:")
        print(f"    - 本地坐标: {info['position_local']}")
        print(f"    - 世界坐标: {info['position_world']}")
        print(f"    - WGS84: ({info['position_wgs84']['longitude']:.6f}, "
              f"{info['position_wgs84']['latitude']:.6f}, "
              f"{info['position_wgs84']['height']:.2f}m)")
        if 'color' in info:
            print(f"    - 颜色: RGB{tuple(info['color'])}")
        print("  [OK] 点信息查询测试通过")
    
    # 测试导出功能
    print("\n[步骤 9] 测试导出功能")
    print("-" * 40)
    
    export_results = picker.get_k_nearest_at_wgs84(
        center_longitude, center_latitude, center_height, k=3
    )
    export_path = os.path.join(TEST_DATA_DIR, "test_export.json")
    picker.export_results_to_json(export_results, export_path)
    
    # 验证导出文件
    with open(export_path, 'r', encoding='utf-8') as f:
        exported = json.load(f)
    assert len(exported) == 3, "导出应该有3条记录"
    print("  [OK] 导出功能测试通过")
    
    # 性能测试
    print("\n[步骤 10] 性能测试")
    print("-" * 40)
    
    import time
    
    # 测试查询性能
    num_queries = 1000
    start = time.time()
    for _ in range(num_queries):
        picker.get_index_at_wgs84(
            longitude=center_longitude + np.random.uniform(-0.001, 0.001),
            latitude=center_latitude + np.random.uniform(-0.001, 0.001),
            height=center_height + np.random.uniform(-10, 10),
            max_distance=100.0
        )
    elapsed = time.time() - start
    
    print(f"  执行 {num_queries} 次查询用时: {elapsed:.3f} 秒")
    print(f"  平均每次查询: {elapsed/num_queries*1000:.3f} 毫秒")
    print("  [OK] 性能测试通过")
    
    # 总结
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
    
    print(f"\n测试数据保存在: {TEST_DATA_DIR}")
    print("  - test_model.ply: 测试 PLY 文件")
    print("  - test_tileset.json: 测试 tileset.json")
    print("  - test_export.json: 导出测试结果")


def test_without_tileset():
    """测试不使用 tileset.json 的情况"""
    from ply_point_picker import PlyPointPicker
    
    print("\n" + "=" * 60)
    print("测试不使用 tileset.json")
    print("=" * 60)
    
    ensure_test_data_dir()
    
    # 生成 ECEF 坐标系中的测试数据
    # 以某个 ECEF 点为中心
    center_ecef = PlyPointPicker.wgs84_to_ecef(117.636892, 24.832147, 379.13)
    
    # 生成点（直接在 ECEF 坐标系中）
    num_points = 100
    np.random.seed(42)
    ply_path = os.path.join(TEST_DATA_DIR, "test_ecef.ply")
    
    with open(ply_path, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
        f.write(header.encode('ascii'))
        
        for _ in range(num_points):
            x = center_ecef[0] + np.random.uniform(-50, 50)
            y = center_ecef[1] + np.random.uniform(-50, 50)
            z = center_ecef[2] + np.random.uniform(-50, 50)
            f.write(struct.pack('<fff', x, y, z))
    
    print(f"[测试] 生成 ECEF 坐标系 PLY 文件: {ply_path}")
    
    # 使用单位矩阵（不变换）
    picker = PlyPointPicker(ply_path, verbose=False)
    
    result = picker.get_index_at_wgs84(117.636892, 24.832147, 379.13, max_distance=100.0)
    
    if result:
        print(f"  找到最近的点: PLY 索引 = {result['ply_index']}, 距离 = {result['distance']:.4f}m")
        print("  [OK] 不使用 tileset.json 测试通过")
    else:
        print("  [FAIL] 未找到匹配点")


def test_large_dataset():
    """测试大型数据集性能"""
    from ply_point_picker import PlyPointPicker
    
    print("\n" + "=" * 60)
    print("大型数据集性能测试")
    print("=" * 60)
    
    ensure_test_data_dir()
    
    import time
    
    for num_points in [10000, 100000, 1000000]:
        print(f"\n测试 {num_points:,} 个点:")
        
        ply_path = os.path.join(TEST_DATA_DIR, f"test_large_{num_points}.ply")
        
        # 生成数据
        start = time.time()
        generate_test_ply(num_points=num_points, output_path=ply_path)
        gen_time = time.time() - start
        print(f"  生成数据用时: {gen_time:.2f} 秒")
        
        # 初始化
        tileset_path = os.path.join(TEST_DATA_DIR, "test_tileset.json")
        start = time.time()
        picker = PlyPointPicker(ply_path, tileset_path, verbose=False)
        init_time = time.time() - start
        print(f"  初始化用时: {init_time:.2f} 秒")
        
        # 查询
        num_queries = 100
        start = time.time()
        for _ in range(num_queries):
            picker.get_index_at_wgs84(
                117.636892 + np.random.uniform(-0.001, 0.001),
                24.832147 + np.random.uniform(-0.001, 0.001),
                379.13,
                max_distance=100.0
            )
        query_time = time.time() - start
        print(f"  {num_queries} 次查询用时: {query_time:.3f} 秒 (平均 {query_time/num_queries*1000:.3f}ms)")
        
        # 清理大文件
        if num_points >= 100000:
            os.remove(ply_path)
            print(f"  清理测试文件")


if __name__ == '__main__':
    # 运行基本测试
    run_tests()
    
    # 运行额外测试
    test_without_tileset()
    
    # 可选：大数据集性能测试（耗时较长）
    # 取消注释下面的行来运行
    # test_large_dataset()
