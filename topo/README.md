# litevloc_topo

关键帧拓扑图上"找最短路、算路点、管哪个目标在哪个关键帧"的一个纯 numpy 小包。原来这件事有两份
代码——nav_system 的 `nav_mapping/map_route.py` 和 LiteVLoc 的 `utils/utils_shortest_path.py`——
现在合成这一份,两边都调它,不再各自维护一套 Dijkstra。

只依赖 `numpy` 和标准库,Python 3.8 语法;不 import rospy / scipy / torch / cv2(`ros_viz.py`
例外,rospy 等消息类型在函数体内延迟 import,不在模块顶层)。

## 装法

```bash
# 两边都要装(各自的解释器各装一次,--no-deps 不带上游依赖):
~/.venvs/nav38/bin/python -m pip install --no-deps -e /home/cmit/robohike_ws/src/opennavmap/third_party/litevloc_code/topo
~/miniconda3/envs/litevloc/bin/python -m pip install --no-deps -e /home/cmit/robohike_ws/src/opennavmap/third_party/litevloc_code/topo
```

跑单测(只需要 numpy + pytest,不需要 ROS):

```bash
cd /home/cmit/robohike_ws/src/opennavmap/third_party/litevloc_code/topo
~/.venvs/nav38/bin/python -m pytest tests -q
```

## 地图目录格式(只读)

- `poses.txt`:每行 `<图片名> qw qx qy qz tx ty tz` = **任务系 → 相机**的变换(四元数 wxyz 顺序
  + 平移);取逆得到相机在任务系(地图系)里的位姿,**行号(从 0 起)就是关键帧编号**,和检索
  服务里的 `kf_NNN` 一致。
- `edges_<edge_type>.txt`:每行 `i j w`,`i`/`j` 是关键帧编号,边无向。边长**默认**按两关键帧
  **机身**平面位置的欧氏距离算,`w` 只在 `use_file_weight=True` 时才用。

机身 vs 相机:相机装在机身中心前方 `camera_forward_m` 米(这是机器人参数,本包不写死,调用方
自己传;nav_system 传 `0.1582`),机身位置 = 相机位置 − `camera_forward_m` × (cos yaw, sin yaw)。

## 五个模块各干什么

- **`graph.py`**:读地图、建 `TopoGraph`(关键帧机身位姿 + 邻接表)。核心函数
  `load_topo_graph(map_dir, camera_forward_m, edge_type="trav", use_file_weight=False)`;
  `TopoGraph` 自带 `.nearest(x, y)`(离某点最近的关键帧)、`.neighbors(kf)`、`.edge_length(i, j)`、
  `.components()`(连通分量,给"这片图连不连通"这类检查用)。`from_edges(poses, edges)` 方便从
  内存数据建图(不用先落盘)。
- **`search.py`**:`dijkstra(graph, start, goal, blocked=())` 最短路(可以传 `blocked` 临时封边,
  按无向处理);`k_hop(graph, seeds, k)` 找多跳邻域内的所有关键帧。全仓库唯一的 Dijkstra 实现。
- **`route.py`**:`sparsify` / `subdivide` 把关键帧序列变成路点序列(太密的中间点丢掉、太远的
  两点间插点);`path_to_waypoints(graph, path)` 把两者串起来;`Route` 数据类装规划结果(关键帧
  序列、总长、路点、每个路点对应哪个关键帧)。
- **`goals.py`**:各种"目标"怎么换成关键帧编号——`KeyframeGoal`(直接给编号)、`ObjectGoal`(按
  物体 id 查 `objects.json` 的 `best_crop`)、`ImageGoal`(按检索排名或图像描述子相似度选)、
  `LanguageGoal`(一句话,交调用方注入的 `resolver` 解析,本包不实现任何文字规则)。
- **`planner.py`**:`TopoPlanner` 把上面几块串成一次任务的规划状态——`plan(start_pose, goal)`、
  `block_edge` / `clear_blocked`(临时封边,一次任务生命周期内持续有效)、`replan`(封边后从新位置
  重规划到上次的目标)、`advance(robot_pose, reach_m)`(逐个弹出走到的子目标,给 LiteVLoc
  `GlobalPlanner` 按里程计推进用)。
- **`ros_viz.py`**(附加,不进核心逻辑):把 `Route` / `TopoGraph` 画成 RViz 消息
  (`nav_msgs/Path`、`visualization_msgs/MarkerArray`),只造消息、不建 publisher。延迟 import
  rospy / ROS 消息类型,纯 numpy 环境不装 ROS 也能 import 这个模块(只是调用里面的函数会报错)。

## 最小例子

```python
from pathlib import Path
from litevloc_topo import load_topo_graph, KeyframeGoal, TopoPlanner, Pose2D

graph = load_topo_graph(Path("/path/to/map"), camera_forward_m=0.1582)
planner = TopoPlanner(graph, map_dir=Path("/path/to/map"))
route = planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(graph.nearest(3.0, 1.0)))
print(route.kf_path, route.length_m)

# 走的过程中按里程计推进子目标:
subgoal_kf = planner.advance(Pose2D(1.0, 0.0, 0.0), reach_m=0.5)

# 发现某条边走不通,临时封掉再重规划:
new_route = planner.replan(Pose2D(1.0, 0.0, 0.0), block_edge=(3, 4))
```
