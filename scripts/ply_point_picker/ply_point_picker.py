"""
PLY 点选择器 - 根据 WGS84 坐标获取 PLY 文件中最近点的索引

使用 KD-Tree 加速大规模点云数据的空间查询，性能可达 O(log n)。

依赖:
    pip install numpy scipy plyfile

使用示例:
    from ply_point_picker import PlyPointPicker
    
    picker = PlyPointPicker("model.ply", "tileset.json")
    result = picker.get_index_at_wgs84(117.636892, 24.832147, 379.13)
    if result:
        print(f"PLY 索引: {result['ply_index']}")
"""

import numpy as np
from scipy.spatial import cKDTree
import json
import os
from typing import Optional, List, Dict, Union, Tuple


class PlyPointPicker:
    """
    PLY 点选择器，使用 KD-Tree 加速空间查询
    
    支持:
    - 根据 WGS84 坐标查询最近的点
    - 查询指定半径内的所有点
    - 查询 K 个最近邻点
    - 批量查询多个坐标点
    """
    
    # WGS84 椭球参数
    WGS84_A = 6378137.0  # 长半轴 (米)
    WGS84_F = 1 / 298.257223563  # 扁率
    WGS84_E2 = 2 * WGS84_F - WGS84_F * WGS84_F  # 第一偏心率的平方
    
    def __init__(
        self,
        ply_file_path: str,
        tileset_json_path: Optional[str] = None,
        root_transform: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        初始化 PLY 点选择器
        
        Args:
            ply_file_path: PLY 文件路径
            tileset_json_path: tileset.json 文件路径（可选，用于获取根变换矩阵）
            root_transform: 4x4 根变换矩阵（可选，优先级高于 tileset_json_path）
            verbose: 是否输出详细信息
        """
        self.ply_file_path = ply_file_path
        self.verbose = verbose
        
        # 读取 PLY 文件
        self._load_ply_file()
        
        # 获取根变换矩阵
        self.root_transform = self._get_root_transform(tileset_json_path, root_transform)
        
        # 将本地坐标转换为世界坐标（ECEF）
        self.positions_world = self._apply_transform(self.positions_local, self.root_transform)
        
        # 构建 KD-Tree
        self._build_kdtree()
        
        if self.verbose:
            print(f"[PlyPointPicker] 初始化完成")
            print(f"  - 点数量: {self.num_points:,}")
            print(f"  - 本地坐标范围: X[{self.positions_local[:, 0].min():.2f}, {self.positions_local[:, 0].max():.2f}]")
            print(f"  - 世界坐标范围: X[{self.positions_world[:, 0].min():.2f}, {self.positions_world[:, 0].max():.2f}]")
    
    def _load_ply_file(self):
        """加载 PLY 文件"""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError("请安装 plyfile: pip install plyfile")
        
        if not os.path.exists(self.ply_file_path):
            raise FileNotFoundError(f"PLY 文件不存在: {self.ply_file_path}")
        
        if self.verbose:
            print(f"[PlyPointPicker] 正在加载 PLY 文件: {self.ply_file_path}")
        
        ply_data = PlyData.read(self.ply_file_path)
        vertex = ply_data['vertex']
        
        # 提取坐标
        self.positions_local = np.vstack([
            vertex['x'],
            vertex['y'],
            vertex['z']
        ]).T.astype(np.float64)
        
        self.num_points = len(self.positions_local)
        
        # 尝试提取颜色（如果存在）
        self.colors = None
        if 'red' in vertex.data.dtype.names:
            self.colors = np.vstack([
                vertex['red'],
                vertex['green'],
                vertex['blue']
            ]).T
        
        if self.verbose:
            print(f"  - 加载了 {self.num_points:,} 个点")
            if self.colors is not None:
                print(f"  - 包含颜色信息")
    
    def _get_root_transform(
        self,
        tileset_json_path: Optional[str],
        root_transform: Optional[np.ndarray]
    ) -> np.ndarray:
        """获取根变换矩阵"""
        # 如果直接提供了变换矩阵，使用它
        if root_transform is not None:
            if self.verbose:
                print(f"[PlyPointPicker] 使用提供的根变换矩阵")
            return np.array(root_transform).reshape(4, 4)
        
        # 如果提供了 tileset.json 路径，从中读取
        if tileset_json_path is not None and os.path.exists(tileset_json_path):
            if self.verbose:
                print(f"[PlyPointPicker] 从 tileset.json 读取根变换矩阵")
            
            with open(tileset_json_path, 'r', encoding='utf-8') as f:
                tileset = json.load(f)
            
            if 'root' in tileset and 'transform' in tileset['root']:
                transform_flat = tileset['root']['transform']
                # 3D Tiles 使用列主序，需要转置
                transform = np.array(transform_flat).reshape(4, 4).T
                if self.verbose:
                    print(f"  - 变换矩阵已加载")
                return transform
        
        # 默认使用单位矩阵
        if self.verbose:
            print(f"[PlyPointPicker] 使用单位矩阵（无变换）")
        return np.eye(4)
    
    def _apply_transform(self, positions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """应用 4x4 变换矩阵到点坐标"""
        # 将位置转换为齐次坐标
        ones = np.ones((positions.shape[0], 1))
        positions_homogeneous = np.hstack([positions, ones])
        
        # 应用变换
        transformed = positions_homogeneous @ matrix.T
        
        return transformed[:, :3]
    
    def _build_kdtree(self):
        """构建 KD-Tree"""
        if self.verbose:
            print(f"[PlyPointPicker] 正在构建 KD-Tree...")
        
        self.tree = cKDTree(self.positions_world)
        
        if self.verbose:
            print(f"  - KD-Tree 构建完成")
    
    @staticmethod
    def wgs84_to_ecef(
        longitude: float,
        latitude: float,
        height: float
    ) -> np.ndarray:
        """
        将 WGS84 坐标转换为 ECEF 笛卡尔坐标
        
        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            height: 高度（米，相对于椭球面）
        
        Returns:
            ECEF 坐标 [x, y, z]
        """
        lon_rad = np.radians(longitude)
        lat_rad = np.radians(latitude)
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # 卯酉圈曲率半径
        N = PlyPointPicker.WGS84_A / np.sqrt(1 - PlyPointPicker.WGS84_E2 * sin_lat * sin_lat)
        
        x = (N + height) * cos_lat * cos_lon
        y = (N + height) * cos_lat * sin_lon
        z = (N * (1 - PlyPointPicker.WGS84_E2) + height) * sin_lat
        
        return np.array([x, y, z])
    
    @staticmethod
    def ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        将 ECEF 笛卡尔坐标转换为 WGS84 坐标
        
        Args:
            x, y, z: ECEF 坐标（米）
        
        Returns:
            (longitude, latitude, height) - 经度（度）、纬度（度）、高度（米）
        """
        a = PlyPointPicker.WGS84_A
        e2 = PlyPointPicker.WGS84_E2
        
        # 计算经度
        longitude = np.degrees(np.arctan2(y, x))
        
        # 迭代计算纬度和高度
        p = np.sqrt(x * x + y * y)
        lat = np.arctan2(z, p * (1 - e2))  # 初始猜测
        
        for _ in range(10):  # 迭代收敛
            sin_lat = np.sin(lat)
            N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
            lat_new = np.arctan2(z + e2 * N * sin_lat, p)
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new
        
        latitude = np.degrees(lat)
        
        # 计算高度
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
        
        if abs(cos_lat) > 1e-10:
            height = p / cos_lat - N
        else:
            height = abs(z) - N * (1 - e2)
        
        return longitude, latitude, height
    
    def get_index_at_wgs84(
        self,
        longitude: float,
        latitude: float,
        height: float,
        max_distance: float = 1.0,
        ignore_height: bool = False,
        height_tolerance: Optional[float] = None
    ) -> Optional[Dict]:
        """
        根据 WGS84 坐标获取最近点的索引
        
        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            height: 高度（米）
            max_distance: 最大搜索距离（米），超过此距离返回 None
            ignore_height: 是否忽略高度进行查询（用于点击穿透场景）
            height_tolerance: 高度容差（米），如果设置，会在指定高度范围内搜索
        
        Returns:
            dict: {
                'ply_index': int,           # PLY 文件中的点索引
                'distance': float,          # 到目标点的距离（米）
                'distance_2d': float,       # 水平距离（米，仅 ignore_height=True 时）
                'position_local': list,     # 本地坐标 [x, y, z]
                'position_world': list,     # 世界坐标 (ECEF) [x, y, z]
                'position_wgs84': dict      # WGS84 坐标 {longitude, latitude, height}
            }
            如果未找到匹配点，返回 None
        """
        if ignore_height:
            # 忽略高度模式：使用 2D KD-Tree 查询
            return self._get_index_ignore_height(longitude, latitude, max_distance)
        
        if height_tolerance is not None:
            # 高度容差模式：在指定高度范围内搜索
            return self._get_index_with_height_tolerance(
                longitude, latitude, height, max_distance, height_tolerance
            )
        
        # 标准 3D 查询
        target_point = self.wgs84_to_ecef(longitude, latitude, height)
        
        # 使用 KD-Tree 查询最近点
        distance, index = self.tree.query(target_point, k=1)
        
        if distance > max_distance:
            return None
        
        # 计算点的 WGS84 坐标
        world_pos = self.positions_world[index]
        wgs84_lon, wgs84_lat, wgs84_h = self.ecef_to_wgs84(*world_pos)
        
        return {
            'ply_index': int(index),
            'distance': float(distance),
            'position_local': self.positions_local[index].tolist(),
            'position_world': world_pos.tolist(),
            'position_wgs84': {
                'longitude': wgs84_lon,
                'latitude': wgs84_lat,
                'height': wgs84_h
            }
        }
    
    def _build_2d_kdtree(self):
        """构建 2D KD-Tree（忽略高度）用于平面查询"""
        if hasattr(self, 'tree_2d'):
            return
        
        if self.verbose:
            print(f"[PlyPointPicker] 正在构建 2D KD-Tree...")
        
        # 将 ECEF 转换为经纬度，只保留经纬度
        self.positions_lonlat = np.zeros((self.num_points, 2))
        for i in range(self.num_points):
            lon, lat, _ = self.ecef_to_wgs84(*self.positions_world[i])
            self.positions_lonlat[i] = [lon, lat]
        
        self.tree_2d = cKDTree(self.positions_lonlat)
        
        if self.verbose:
            print(f"  - 2D KD-Tree 构建完成")
    
    def _get_index_ignore_height(
        self,
        longitude: float,
        latitude: float,
        max_distance: float
    ) -> Optional[Dict]:
        """
        忽略高度查询最近点（用于点击穿透场景）
        
        注意：max_distance 在这里是经纬度的"度"距离，不是米
        为了方便使用，会自动转换：1度 ≈ 111km
        """
        # 确保 2D KD-Tree 已构建
        self._build_2d_kdtree()
        
        # 将米转换为度（近似）：1度 ≈ 111000米
        max_distance_deg = max_distance / 111000.0
        
        # 使用 2D KD-Tree 查询
        target_2d = np.array([longitude, latitude])
        distance_deg, index = self.tree_2d.query(target_2d, k=1)
        
        if distance_deg > max_distance_deg:
            return None
        
        # 计算实际水平距离（米）
        distance_2d_meters = distance_deg * 111000.0
        
        # 计算 3D 距离
        world_pos = self.positions_world[index]
        wgs84_lon, wgs84_lat, wgs84_h = self.ecef_to_wgs84(*world_pos)
        
        return {
            'ply_index': int(index),
            'distance': float(distance_2d_meters),  # 水平距离
            'distance_2d': float(distance_2d_meters),
            'position_local': self.positions_local[index].tolist(),
            'position_world': world_pos.tolist(),
            'position_wgs84': {
                'longitude': wgs84_lon,
                'latitude': wgs84_lat,
                'height': wgs84_h
            }
        }
    
    def _get_index_with_height_tolerance(
        self,
        longitude: float,
        latitude: float,
        height: float,
        max_distance: float,
        height_tolerance: float
    ) -> Optional[Dict]:
        """
        在指定高度范围内查询最近点
        
        先用较大范围查询候选点，然后筛选高度在容差范围内的点
        """
        target_point = self.wgs84_to_ecef(longitude, latitude, height)
        
        # 使用较大的搜索范围获取候选点
        search_radius = max(max_distance, height_tolerance) * 2
        indices = self.tree.query_ball_point(target_point, search_radius)
        
        if not indices:
            return None
        
        # 筛选高度在容差范围内的点
        best_result = None
        best_distance = float('inf')
        
        for idx in indices:
            world_pos = self.positions_world[idx]
            _, _, point_height = self.ecef_to_wgs84(*world_pos)
            
            # 检查高度是否在容差范围内
            height_diff = abs(point_height - height)
            if height_diff > height_tolerance:
                continue
            
            # 计算水平距离
            point_lon, point_lat, _ = self.ecef_to_wgs84(*world_pos)
            horizontal_distance = self._haversine_distance(
                longitude, latitude, point_lon, point_lat
            )
            
            if horizontal_distance <= max_distance and horizontal_distance < best_distance:
                best_distance = horizontal_distance
                best_result = {
                    'ply_index': int(idx),
                    'distance': float(horizontal_distance),
                    'distance_2d': float(horizontal_distance),
                    'height_diff': float(height_diff),
                    'position_local': self.positions_local[idx].tolist(),
                    'position_world': world_pos.tolist(),
                    'position_wgs84': {
                        'longitude': point_lon,
                        'latitude': point_lat,
                        'height': point_height
                    }
                }
        
        return best_result
    
    @staticmethod
    def _haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        计算两个经纬度点之间的地表距离（米）
        使用 Haversine 公式
        """
        R = 6371000  # 地球半径（米）
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat / 2) ** 2 + \
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def get_k_nearest_at_wgs84(
        self,
        longitude: float,
        latitude: float,
        height: float,
        k: int = 5,
        max_distance: Optional[float] = None
    ) -> List[Dict]:
        """
        获取 K 个最近邻点
        
        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            height: 高度（米）
            k: 返回的最近邻数量
            max_distance: 最大搜索距离（米），可选
        
        Returns:
            按距离排序的结果列表
        """
        target_point = self.wgs84_to_ecef(longitude, latitude, height)
        
        # 查询 K 个最近邻
        distances, indices = self.tree.query(target_point, k=k)
        
        # 确保 distances 和 indices 是数组
        if k == 1:
            distances = [distances]
            indices = [indices]
        
        results = []
        for dist, idx in zip(distances, indices):
            if max_distance is not None and dist > max_distance:
                continue
            
            world_pos = self.positions_world[idx]
            wgs84_lon, wgs84_lat, wgs84_h = self.ecef_to_wgs84(*world_pos)
            
            results.append({
                'ply_index': int(idx),
                'distance': float(dist),
                'position_local': self.positions_local[idx].tolist(),
                'position_world': world_pos.tolist(),
                'position_wgs84': {
                    'longitude': wgs84_lon,
                    'latitude': wgs84_lat,
                    'height': wgs84_h
                }
            })
        
        return results
    
    def get_indices_in_radius(
        self,
        longitude: float,
        latitude: float,
        height: float,
        radius: float
    ) -> List[Dict]:
        """
        获取指定半径内所有点的索引
        
        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            height: 高度（米）
            radius: 搜索半径（米）
        
        Returns:
            按距离排序的结果列表
        """
        target_point = self.wgs84_to_ecef(longitude, latitude, height)
        
        # 查询半径内的所有点
        indices = self.tree.query_ball_point(target_point, radius)
        
        results = []
        for idx in indices:
            distance = np.linalg.norm(self.positions_world[idx] - target_point)
            world_pos = self.positions_world[idx]
            wgs84_lon, wgs84_lat, wgs84_h = self.ecef_to_wgs84(*world_pos)
            
            results.append({
                'ply_index': int(idx),
                'distance': float(distance),
                'position_local': self.positions_local[idx].tolist(),
                'position_world': world_pos.tolist(),
                'position_wgs84': {
                    'longitude': wgs84_lon,
                    'latitude': wgs84_lat,
                    'height': wgs84_h
                }
            })
        
        return sorted(results, key=lambda x: x['distance'])
    
    def batch_query(
        self,
        coordinates: List[Tuple[float, float, float]],
        max_distance: float = 1.0
    ) -> List[Optional[Dict]]:
        """
        批量查询多个坐标点
        
        Args:
            coordinates: 坐标列表 [(longitude, latitude, height), ...]
            max_distance: 最大搜索距离（米）
        
        Returns:
            结果列表，每个元素对应输入的一个坐标点
        """
        results = []
        for lon, lat, h in coordinates:
            result = self.get_index_at_wgs84(lon, lat, h, max_distance)
            results.append(result)
        return results
    
    def get_point_info(self, ply_index: int) -> Optional[Dict]:
        """
        根据 PLY 索引获取点的详细信息
        
        Args:
            ply_index: PLY 文件中的点索引
        
        Returns:
            点的详细信息
        """
        if ply_index < 0 or ply_index >= self.num_points:
            return None
        
        world_pos = self.positions_world[ply_index]
        wgs84_lon, wgs84_lat, wgs84_h = self.ecef_to_wgs84(*world_pos)
        
        result = {
            'ply_index': ply_index,
            'position_local': self.positions_local[ply_index].tolist(),
            'position_world': world_pos.tolist(),
            'position_wgs84': {
                'longitude': wgs84_lon,
                'latitude': wgs84_lat,
                'height': wgs84_h
            }
        }
        
        if self.colors is not None:
            result['color'] = self.colors[ply_index].tolist()
        
        return result
    
    def export_results_to_json(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        将查询结果导出为 JSON 文件
        
        Args:
            results: 查询结果列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"[PlyPointPicker] 结果已导出到: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='根据 WGS84 坐标查询 PLY 文件中最近的点'
    )
    parser.add_argument('ply_file', help='PLY 文件路径')
    parser.add_argument('--tileset', help='tileset.json 文件路径')
    parser.add_argument('--lon', type=float, required=True, help='经度（度）')
    parser.add_argument('--lat', type=float, required=True, help='纬度（度）')
    parser.add_argument('--height', type=float, default=0.0, help='高度（米），默认 0')
    parser.add_argument('--max-distance', type=float, default=1.0, help='最大搜索距离（米）')
    parser.add_argument('--k', type=int, default=1, help='返回的最近邻数量')
    parser.add_argument('--ignore-height', action='store_true', 
                        help='忽略高度进行查询（适用于点击穿透场景）')
    parser.add_argument('--height-tolerance', type=float, default=None,
                        help='高度容差（米），在指定高度范围内搜索')
    parser.add_argument('--output', help='输出 JSON 文件路径')
    
    args = parser.parse_args()
    
    # 创建选择器
    picker = PlyPointPicker(args.ply_file, args.tileset)
    
    # 查询
    if args.k == 1:
        result = picker.get_index_at_wgs84(
            args.lon, args.lat, args.height, args.max_distance,
            ignore_height=args.ignore_height,
            height_tolerance=args.height_tolerance
        )
        results = [result] if result else []
    else:
        results = picker.get_k_nearest_at_wgs84(
            args.lon, args.lat, args.height, args.k, args.max_distance
        )
    
    # 输出结果
    if results:
        print(f"\n找到 {len(results)} 个匹配点:")
        for i, r in enumerate(results):
            print(f"\n  [{i + 1}] PLY 索引: {r['ply_index']}")
            print(f"      距离: {r['distance']:.4f} 米")
            if 'distance_2d' in r:
                print(f"      水平距离: {r['distance_2d']:.4f} 米")
            if 'height_diff' in r:
                print(f"      高度差: {r['height_diff']:.4f} 米")
            print(f"      本地坐标: {r['position_local']}")
            print(f"      WGS84: ({r['position_wgs84']['longitude']:.6f}°, "
                  f"{r['position_wgs84']['latitude']:.6f}°, "
                  f"{r['position_wgs84']['height']:.2f}m)")
        
        # 导出到文件
        if args.output:
            picker.export_results_to_json(results, args.output)
    else:
        print("\n未找到匹配的点")


if __name__ == '__main__':
    main()
