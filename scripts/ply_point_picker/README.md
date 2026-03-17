# PLY 点选择器 (PLY Point Picker)

根据 WGS84 坐标查询 PLY 文件中最近点的索引，使用 KD-Tree 加速空间查询。

## 功能特点

- **高性能**: 使用 KD-Tree 实现 O(log n) 的查询复杂度
- **坐标转换**: 自动处理 WGS84 ↔ ECEF 坐标转换
- **模型变换**: 支持从 tileset.json 读取根变换矩阵
- **多种查询方式**:
  - 最近点查询
  - K 最近邻查询
  - 半径范围查询
  - 批量坐标查询

## 安装依赖

```bash
pip install numpy scipy plyfile
```

## 快速开始

### Python API 使用

```python
from ply_point_picker import PlyPointPicker

# 初始化选择器
picker = PlyPointPicker(
    ply_file_path="model.ply",
    tileset_json_path="tileset.json"  # 可选
)

# 根据 WGS84 坐标查询最近的点
result = picker.get_index_at_wgs84(
    longitude=117.636892,
    latitude=24.832147,
    height=379.13,
    max_distance=1.0  # 最大搜索距离（米）
)

if result:
    print(f"PLY 索引: {result['ply_index']}")
    print(f"距离: {result['distance']:.4f} 米")
    print(f"WGS84 坐标: {result['position_wgs84']}")
```

### 命令行使用

```bash
# 基本查询
python ply_point_picker.py model.ply --lon 117.636892 --lat 24.832147 --height 379.13

# 使用 tileset.json
python ply_point_picker.py model.ply --tileset tileset.json --lon 117.636892 --lat 24.832147 --height 379.13

# 查询 K 个最近邻
python ply_point_picker.py model.ply --lon 117.636892 --lat 24.832147 --height 379.13 --k 5

# 忽略高度查询（适用于 Cesium 点击穿透场景）
python ply_point_picker.py model.ply --lon 117.636892 --lat 24.832147 --ignore-height --max-distance 10

# 指定高度容差查询（在 ±50 米范围内搜索）
python ply_point_picker.py model.ply --lon 117.636892 --lat 24.832147 --height 379.13 --height-tolerance 50

# 输出到 JSON 文件
python ply_point_picker.py model.ply --lon 117.636892 --lat 24.832147 --height 379.13 --output result.json
```

## API 参考

### PlyPointPicker 类

#### 构造函数

```python
PlyPointPicker(
    ply_file_path: str,
    tileset_json_path: Optional[str] = None,
    root_transform: Optional[np.ndarray] = None,
    verbose: bool = True
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ply_file_path` | str | PLY 文件路径 |
| `tileset_json_path` | str, 可选 | tileset.json 文件路径，用于获取根变换矩阵 |
| `root_transform` | np.ndarray, 可选 | 4x4 根变换矩阵，优先级高于 tileset_json_path |
| `verbose` | bool | 是否输出详细信息 |

#### 方法

##### `get_index_at_wgs84()`

根据 WGS84 坐标获取最近点的索引。

```python
get_index_at_wgs84(
    longitude: float,
    latitude: float,
    height: float,
    max_distance: float = 1.0,
    ignore_height: bool = False,      # 忽略高度查询（适用于点击穿透）
    height_tolerance: float = None    # 高度容差（米）
) -> Optional[Dict]
```

**参数说明**:
- `ignore_height`: 设为 `True` 时只用经纬度做 2D 查询，忽略高度差异。适用于 Cesium 点击穿透场景，射线穿过模型后返回的高度不准确的情况。
- `height_tolerance`: 设置后会在指定高度范围内搜索（如 `50` 表示 ±50 米），同时返回水平距离最近的点。

**返回值**:
```python
{
    'ply_index': int,           # PLY 文件中的点索引
    'distance': float,          # 到目标点的距离（米）
    'distance_2d': float,       # 水平距离（米，仅 ignore_height 或 height_tolerance 模式）
    'height_diff': float,       # 高度差（米，仅 height_tolerance 模式）
    'position_local': list,     # 本地坐标 [x, y, z]
    'position_world': list,     # 世界坐标 (ECEF) [x, y, z]
    'position_wgs84': {         # WGS84 坐标
        'longitude': float,
        'latitude': float,
        'height': float
    }
}
```

##### `get_k_nearest_at_wgs84()`

获取 K 个最近邻点。

```python
get_k_nearest_at_wgs84(
    longitude: float,
    latitude: float,
    height: float,
    k: int = 5,
    max_distance: Optional[float] = None
) -> List[Dict]
```

##### `get_indices_in_radius()`

获取指定半径内所有点的索引。

```python
get_indices_in_radius(
    longitude: float,
    latitude: float,
    height: float,
    radius: float
) -> List[Dict]
```

##### `batch_query()`

批量查询多个坐标点。

```python
batch_query(
    coordinates: List[Tuple[float, float, float]],
    max_distance: float = 1.0
) -> List[Optional[Dict]]
```

##### `get_point_info()`

根据 PLY 索引获取点的详细信息。

```python
get_point_info(ply_index: int) -> Optional[Dict]
```

##### `export_results_to_json()`

将查询结果导出为 JSON 文件。

```python
export_results_to_json(results: List[Dict], output_path: str)
```

#### 静态方法

##### `wgs84_to_ecef()`

将 WGS84 坐标转换为 ECEF 笛卡尔坐标。

```python
@staticmethod
wgs84_to_ecef(longitude: float, latitude: float, height: float) -> np.ndarray
```

##### `ecef_to_wgs84()`

将 ECEF 笛卡尔坐标转换为 WGS84 坐标。

```python
@staticmethod
ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]
```

## 使用示例

### 示例 1: 基本查询

```python
from ply_point_picker import PlyPointPicker

# 加载 PLY 文件
picker = PlyPointPicker("model.ply")

# 查询单个点
result = picker.get_index_at_wgs84(117.636892, 24.832147, 379.13)
if result:
    print(f"找到最近的点: PLY 索引 = {result['ply_index']}")
```

### 示例 2: 使用 tileset.json

```python
picker = PlyPointPicker(
    ply_file_path="model.ply",
    tileset_json_path="tileset.json"
)

result = picker.get_index_at_wgs84(117.636892, 24.832147, 379.13)
```

### 示例 3: 查询半径内所有点

```python
picker = PlyPointPicker("model.ply")

# 获取 0.5 米半径内的所有点
results = picker.get_indices_in_radius(
    longitude=117.636892,
    latitude=24.832147,
    height=379.13,
    radius=0.5
)

print(f"在 0.5m 半径内找到 {len(results)} 个点")
for r in results[:5]:
    print(f"  PLY 索引 = {r['ply_index']}, 距离 = {r['distance']:.4f}m")
```

### 示例 4: 批量查询

```python
picker = PlyPointPicker("model.ply")

# 准备多个查询坐标
coordinates = [
    (117.636892, 24.832147, 379.13),
    (117.636900, 24.832150, 379.15),
    (117.636880, 24.832140, 379.10),
]

# 批量查询
results = picker.batch_query(coordinates, max_distance=1.0)

for i, (coord, result) in enumerate(zip(coordinates, results)):
    if result:
        print(f"坐标 {i+1}: PLY 索引 = {result['ply_index']}")
    else:
        print(f"坐标 {i+1}: 未找到匹配点")
```

### 示例 5: K 最近邻查询

```python
picker = PlyPointPicker("model.ply")

# 获取 5 个最近邻
results = picker.get_k_nearest_at_wgs84(
    longitude=117.636892,
    latitude=24.832147,
    height=379.13,
    k=5
)

print("5 个最近邻点:")
for i, r in enumerate(results):
    print(f"  [{i+1}] PLY 索引 = {r['ply_index']}, 距离 = {r['distance']:.4f}m")
```

### 示例 6: 导出结果到 JSON

```python
picker = PlyPointPicker("model.ply")

# 查询并导出
results = picker.get_indices_in_radius(117.636892, 24.832147, 379.13, radius=1.0)
picker.export_results_to_json(results, "query_results.json")
```

### 示例 7: Cesium 点击穿透场景（忽略高度）

当在 Cesium 中点击模型时，射线可能穿透模型到达地面，导致返回的高度不准确。
此时可以使用 `ignore_height=True` 只根据经纬度查询：

```python
picker = PlyPointPicker("model.ply", "tileset.json")

# Cesium 点击返回的坐标（高度可能不准确）
click_lon = 117.636892
click_lat = 24.832147
click_height = 0  # 高度可能是地面高度

# 忽略高度，只用经纬度查询
result = picker.get_index_at_wgs84(
    longitude=click_lon,
    latitude=click_lat,
    height=click_height,
    max_distance=10.0,      # 水平距离 10 米内
    ignore_height=True      # 忽略高度
)

if result:
    print(f"PLY 索引: {result['ply_index']}")
    print(f"水平距离: {result['distance_2d']:.4f} 米")
    print(f"实际高度: {result['position_wgs84']['height']:.2f} 米")
```

### 示例 8: 指定高度范围查询

如果你知道大概高度范围，可以使用 `height_tolerance` 限制搜索范围：

```python
picker = PlyPointPicker("model.ply", "tileset.json")

# 在 ±100 米高度范围内搜索
result = picker.get_index_at_wgs84(
    longitude=117.636892,
    latitude=24.832147,
    height=379.13,
    max_distance=5.0,         # 水平距离 5 米内
    height_tolerance=100.0    # 高度容差 ±100 米
)

if result:
    print(f"PLY 索引: {result['ply_index']}")
    print(f"水平距离: {result['distance_2d']:.4f} 米")
    print(f"高度差: {result['height_diff']:.4f} 米")
```

## 坐标系统说明

### WGS84 坐标系

- **经度 (longitude)**: -180° 到 +180°，东经为正
- **纬度 (latitude)**: -90° 到 +90°，北纬为正
- **高度 (height)**: 相对于 WGS84 椭球面的高度，单位为米

### ECEF 坐标系 (Earth-Centered, Earth-Fixed)

- **X 轴**: 指向本初子午线与赤道的交点
- **Y 轴**: 指向东经 90° 与赤道的交点
- **Z 轴**: 指向北极

### 本地坐标系

PLY 文件中的坐标通常是本地坐标系，需要通过模型矩阵转换到 ECEF 坐标系。

## 性能说明

| 数据集大小 | 初始化时间 | 单次查询时间 |
|-----------|-----------|-------------|
| 10,000 点 | < 0.1 秒 | < 1 毫秒 |
| 100,000 点 | < 0.5 秒 | < 1 毫秒 |
| 1,000,000 点 | < 2 秒 | < 1 毫秒 |
| 10,000,000 点 | < 20 秒 | < 2 毫秒 |

KD-Tree 的查询复杂度为 O(log n)，几乎不受数据集大小影响。

## 与 Cesium getSplatInfoAtScreenPosition 的对比

| 特性 | Python PlyPointPicker | Cesium getSplatInfoAtScreenPosition |
|-----|----------------------|-------------------------------------|
| 输入 | WGS84 坐标 | 屏幕像素坐标 |
| 查询方式 | KD-Tree 空间索引 | CPU 线性搜索 / GPU 拾取 |
| 性能 | O(log n) | O(n) (CPU) / O(1) (GPU) |
| 环境 | 离线 Python 脚本 | 浏览器实时交互 |
| 适用场景 | 批量处理、数据分析 | 实时点击拾取 |

## 注意事项

1. **坐标系统**: 确保 PLY 文件的坐标系与你的理解一致
2. **模型矩阵**: 使用 3D Tiles 时，需要应用 tileset.json 中的 `root.transform`
3. **精度**: WGS84 到 ECEF 的转换使用标准椭球参数
4. **内存**: 大型点云会占用较多内存（每个点约 24 字节用于坐标存储）

## 常见问题

### Q: 为什么查询结果的距离很大？

A: 可能是因为没有正确应用模型矩阵。检查：
1. 是否提供了 tileset.json 路径
2. tileset.json 中是否有 `root.transform`
3. PLY 文件的坐标系是否正确

### Q: Cesium 点击时高度穿透怎么办？

A: 在 Cesium 中点击模型时，射线可能穿透模型获取到地面高度。解决方案：

**方案一：忽略高度查询**
```python
result = picker.get_index_at_wgs84(
    lon, lat, height,
    max_distance=10.0,
    ignore_height=True  # 只用经纬度查询
)
```

**方案二：指定高度容差**
```python
result = picker.get_index_at_wgs84(
    lon, lat, height,
    max_distance=5.0,
    height_tolerance=100.0  # 在 ±100 米范围内搜索
)
```

**方案三：在 Cesium 端获取正确高度**
```javascript
// 使用 scene.pickPosition 获取更准确的位置
const cartesian = viewer.scene.pickPosition(windowPosition);
const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
const lon = Cesium.Math.toDegrees(cartographic.longitude);
const lat = Cesium.Math.toDegrees(cartographic.latitude);
const height = cartographic.height;
```

### Q: 如何处理多个 PLY 文件？

A: 为每个 PLY 文件创建一个 PlyPointPicker 实例，然后在查询时遍历所有实例。

### Q: 如何提高初始化速度？

A: 对于大型数据集，可以考虑：
1. 预计算并缓存 KD-Tree
2. 使用 pickle 序列化 PlyPointPicker 对象
3. 只加载需要的区域数据

## 文件结构

```
ply_point_picker/
├── ply_point_picker.py    # 主模块
├── README.md              # 说明文档
├── test_ply_point_picker.py  # 测试脚本
└── test_data/             # 测试数据
    ├── test_model.ply     # 测试 PLY 文件
    └── test_tileset.json  # 测试 tileset.json
```

## License

MIT License
