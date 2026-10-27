"""统一的几何核验口径(M5-B4)。

背景:核验一个候选关键帧对不对,以前是直接拿匹配器自己算的"单应矩阵内点数"当分数——但室内场景
不是一个平面,单应假设本来就不成立,而且这个数会随匹配器是稠密(MASt3R/EDM)还是稀疏(xfeat)
疏密不同,根本没法跨匹配器比。横评(B3)挑中 EDM 之后,这个"核验通过但实际解出来的位姿错得多"
的问题更明显,所以改成一把尺子量到底:

    核验分数 = 查询图上按像素网格去重(避免密集匹配器几千个挤在一小片、单纯靠数量堆分)后,
               用查询深度把这些点撑成 3D,再跟地图图像点解一次 PnP-RANSAC,内点数就是分数。

这把尺子只认"查询深度 + PnP 解出来的点对不对得上",跟用的是哪个匹配器、匹配器自己内部筛不筛
单应内点都无关。查询没有深度时(比如拿一张照片当目标,不是跟着机器人走的那种有深度的观测),
没法做 PnP,退一步用基础矩阵(对极几何约束)内点数打个分,只能用来"认出是哪个关键帧",
不能解出位姿。

本文件是纯几何代码,不 import rospy,方便离线、在线、单测三处共用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 网格去重默认参数:16px 一格、最多留 1500 个点。写成模块常量方便测试里复用同一份默认值。
DEFAULT_CELL_PX = 16
DEFAULT_MAX_POINTS = 1500


def grid_dedup(
    kpts_query: np.ndarray,
    kpts_map: np.ndarray,
    cell_px: int = DEFAULT_CELL_PX,
    max_points: int = DEFAULT_MAX_POINTS,
    scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """按查询图像素网格(cell_px x cell_px)去重:每格只留一个匹配,最多留 max_points 个。

    这是为了不让稠密匹配器(比如 MASt3R/EDM,几千个匹配挤在同一小片纹理丰富的区域)单纯靠
    堆数量在后面的 PnP-RANSAC 里占便宜——跟稀疏匹配器(比如 xfeat,每张图就几百个关键点,
    天然分布在整张图上)比内点数才公平。

    Args:
        kpts_query: (N, 2) 查询图里的匹配点像素坐标(x, y)。
        kpts_map: (N, 2) 对应的地图图里的匹配点像素坐标,跟 kpts_query 逐行一一对应。
        cell_px: 网格边长,单位像素。
        max_points: 去重后最多保留的点数,多出的按网格扫描到的顺序截断(不影响正确性,
            只是给极端情况加个硬上限,避免 PnP-RANSAC 跑太久)。
        scores: 每个匹配的置信度分数(越大越好)。给了就同一格里留分数最高的;不给就留
            第一个扫到的(大多数匹配器不返回逐点置信度,这时退化成"先到先得")。

    Returns:
        去重后的 (kpts_query, kpts_map),两者行数相同、逐行对应;没有匹配点时返回原样的空数组。
    """
    n = kpts_query.shape[0]
    if n == 0:
        return kpts_query.copy(), kpts_map.copy()

    # 用网格坐标 (cx, cy) 编码成一个整数键做字典去重:比 Python tuple 当 key 快,
    # 字典插入顺序又是稳定的(Python 3.7+),方便"先到先得"时结果可复现。
    cells = np.floor(kpts_query / float(cell_px)).astype(np.int64)
    keys = cells[:, 0] * 1_000_000 + cells[:, 1]

    best_idx_by_cell: Dict[int, int] = {}
    for i in range(n):
        key = int(keys[i])
        if key not in best_idx_by_cell:
            best_idx_by_cell[key] = i
        elif scores is not None and scores[i] > scores[best_idx_by_cell[key]]:
            best_idx_by_cell[key] = i

    kept = list(best_idx_by_cell.values())
    if len(kept) > max_points:
        kept = kept[:max_points]

    return kpts_query[kept], kpts_map[kept]


@dataclass
class VerifyResult:
    """一次核验的结果。

    inliers: 统一核验分数(网格去重后的 PnP-RANSAC 内点数),跟 gv_min_inliers 比大小用这个。
    T_world_query: 解出来的查询相机在世界(地图)系下的位姿(4x4,相机到世界);PnP 没解出
        有效位姿(点太少、RANSAC 失败)时是 None —— 这时 inliers 仍然是有效分数(可能是 0),
        调用方该怎么处理由自己决定,这个函数不替调用方做"没位姿就当失败"的判断。
    """

    inliers: int
    T_world_query: Optional[np.ndarray]


def verify_pnp(
    kpts_query: np.ndarray,
    kpts_map: np.ndarray,
    depth_query: np.ndarray,
    K_query: np.ndarray,
    K_map: np.ndarray,
    T_world_map_cam: np.ndarray,
    solver,
    cell_px: int = DEFAULT_CELL_PX,
    max_points: int = DEFAULT_MAX_POINTS,
) -> VerifyResult:
    """网格去重 + 查询深度反投影成 3D + 对地图图像点解 PnP-RANSAC,内点数当核验分数。

    跟局部定位用的是同一个 solver(同一套 RANSAC 参数),这里不重复定义 PnP 逻辑,只是在喂给
    它之前先做统一的网格去重。

    Args:
        kpts_query: (N, 2) 查询图里的匹配点像素坐标,原始(未缩放)分辨率下的坐标。
        kpts_map: (N, 2) 对应的地图图里的匹配点像素坐标,跟 kpts_query 逐行一一对应,
            同样是原始分辨率下的坐标。
        depth_query: 查询图的深度图(米,HxW),给 kpts_query 撑出 3D 点用。
        K_query: 查询相机内参(3x3,原始分辨率)。
        K_map: 地图关键帧相机内参(3x3,原始分辨率)。
        T_world_map_cam: 地图关键帧相机在世界(地图)系下的位姿(4x4,相机到世界的变换,
            即该关键帧的 trans/quat 换成的齐次矩阵)。
        solver: 复用现有 `utils.pose_solver` 的求解器(和局部定位同一个实例/同一套参数),
            只用它的 `estimate_pose(pts0, pts1, K0, K1, depth0, depth1)` 接口。

    Returns:
        VerifyResult:匹配点去重后不足 4 个、或 PnP-RANSAC 没解出来时,inliers 为实际内点数
        (可能是 0),T_world_query 为 None。
    """
    kq, km = grid_dedup(kpts_query, kpts_map, cell_px=cell_px, max_points=max_points)
    if kq.shape[0] < 4:
        return VerifyResult(inliers=0, T_world_query=None)

    # solver.estimate_pose 的第一组(pts0/K0/depth0)必须是同一个相机的;这里是查询相机,
    # 因为只有查询图有深度(局部定位 perform_local_loc 用的也是这个顺序,见 pose_solver.py)。
    R, t, n_inliers = solver.estimate_pose(kq, km, K_query, K_map, depth_query, None)
    n_inliers = int(n_inliers)
    if n_inliers <= 0 or R is None or not np.all(np.isfinite(R)):
        return VerifyResult(inliers=n_inliers, T_world_query=None)

    T_mapcam_querycam = np.eye(4)
    T_mapcam_querycam[:3, :3] = R
    T_mapcam_querycam[:3, 3] = np.asarray(t).reshape(3)
    T_world_query = T_world_map_cam @ T_mapcam_querycam
    return VerifyResult(inliers=n_inliers, T_world_query=T_world_query)


def verify_match_result(
    match_result: Dict[str, np.ndarray],
    map_raw_K: np.ndarray,
    map_raw_img_size: Tuple[int, int],
    map_img_size: Tuple[int, int],
    query_raw_K: np.ndarray,
    query_raw_img_size: Tuple[int, int],
    query_img_size: Tuple[int, int],
    T_world_map_cam: np.ndarray,
    depth_query: Optional[np.ndarray],
    solver,
    cell_px: int = DEFAULT_CELL_PX,
    max_points: int = DEFAULT_MAX_POINTS,
) -> VerifyResult:
    """给匹配器已经跑完的一对结果(`match_result['inlier_kpts0']` 是地图图里的点,
    `match_result['inlier_kpts1']` 是查询图里的点,都是喂进匹配器之前缩放后的分辨率下的像素坐标)
    做统一核验:先换算回各自原始分辨率,再按有没有查询深度走 PnP(verify_pnp)或基础矩阵
    (verify_fundamental,这时解不出位姿,T_world_query 恒为 None)。

    这是 `LocPipeline._verify_one_candidate`(在线/ROS 相关的 `loc_pipeline.py`)和
    `run_vloc_offline_rerun.py`(纯离线重跑,不能 import rospy)两边共用的同一份核验逻辑——
    `loc_pipeline.py` 顶部 import 了 rospy/tf2_ros/nav_msgs,run_vloc_offline_rerun.py 的运行
    环境(`setup_litevloc_env.sh` 建的 litevloc conda 环境)故意把 ROS 从 PYTHONPATH 里剔掉了
    (见 `start_vloc_offline.sh` 的注释),没法 import loc_pipeline.LocPipeline,所以两边各自
    持有一份"图匹配结果 -> 核验分数/位姿"的调用,但都转发到这同一个函数,不是两份独立实现。

    Args:
        match_result: 匹配器 `__call__`/`forward()` 的原始返回值,至少要有
            `inlier_kpts0`/`inlier_kpts1`。
        map_raw_K / map_raw_img_size / map_img_size: 地图关键帧的原始内参、原始分辨率、
            喂进匹配器前缩放到的分辨率。
        query_raw_K / query_raw_img_size / query_img_size: 查询帧的同上三项。
        T_world_map_cam: 地图关键帧相机在世界(地图)系下的位姿(4x4)。
        depth_query: 查询帧的深度图(米);没有深度传 None,会退化成基础矩阵核验
            (只给"认出哪个关键帧"的分数,不解位姿)。
        solver: 复用的 PnP 求解器(utils.pose_solver)。

    Returns:
        VerifyResult。
    """
    mkpts0 = match_result["inlier_kpts0"]
    mkpts1 = match_result["inlier_kpts1"]
    scale_map = np.array([map_raw_img_size[0] / map_img_size[0], map_raw_img_size[1] / map_img_size[1]])
    scale_query = np.array([query_raw_img_size[0] / query_img_size[0], query_raw_img_size[1] / query_img_size[1]])
    mkpts0_raw = mkpts0 * scale_map
    mkpts1_raw = mkpts1 * scale_query

    if depth_query is not None:
        return verify_pnp(
            mkpts1_raw, mkpts0_raw, depth_query,
            query_raw_K, map_raw_K, T_world_map_cam, solver,
            cell_px=cell_px, max_points=max_points,
        )

    inliers = verify_fundamental(mkpts1_raw, mkpts0_raw, cell_px=cell_px, max_points=max_points)
    return VerifyResult(inliers=inliers, T_world_query=None)


def verify_fundamental(
    kpts_query: np.ndarray,
    kpts_map: np.ndarray,
    cell_px: int = DEFAULT_CELL_PX,
    max_points: int = DEFAULT_MAX_POINTS,
    reproj_thresh_px: float = 1.0,
) -> int:
    """查询没有深度时(比如照片目标)没法解 PnP,退一步只看匹配点满不满足对极几何约束
    (基础矩阵),只用来打一个"像不像这个关键帧"的分数,不解位姿。

    Returns:
        网格去重后,基础矩阵 RANSAC(USAC_MAGSAC,阈值 1 像素)的内点数;点不够(< 8,
        基础矩阵最少需要的对应点数)、解不出来、或者 OpenCV 自己内部算不出来直接报错
        (某些点集会让 ``cv2.findFundamentalMat`` 抛 ``cv2.error``,比如实测报过的
        ``!model.empty() in function 'setModelParameters'``)时都返回 0,跟"解不出来"同一个
        处理,不往上抛。
    """
    kq, km = grid_dedup(kpts_query, kpts_map, cell_px=cell_px, max_points=max_points)
    if kq.shape[0] < 8:
        return 0
    try:
        _, mask = cv2.findFundamentalMat(kq, km, cv2.USAC_MAGSAC, reproj_thresh_px)
    except cv2.error as exc:
        logger.warning("基础矩阵求解抛异常,按解不出来处理(内点数记 0): %s", exc)
        return 0
    if mask is None:
        return 0
    return int(mask.sum())
