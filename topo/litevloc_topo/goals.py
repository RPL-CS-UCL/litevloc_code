"""目标 → 关键帧编号。"""

import json
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

from .graph import TopoGraph, TopoGraphError

OBJECTS_FILENAME = "objects.json"


class Goal(Protocol):
    """"目标"的统一接口:给定图(和可选的地图目录)解出一个具体的关键帧编号。"""

    def resolve(self, graph: TopoGraph, map_dir: Optional[Path]) -> int:
        ...


class KeyframeGoal:
    """直接给关键帧编号的目标。"""

    def __init__(self, kf_id: int) -> None:
        self.kf_id = kf_id

    def resolve(self, graph: TopoGraph, map_dir: Optional[Path]) -> int:
        return self.kf_id


class ObjectGoal:
    """按物体实例 id 去 ``objects.json`` 查"看得最清楚"的关键帧(``best_crop`` 第一项)。"""

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id

    def resolve(self, graph: TopoGraph, map_dir: Optional[Path]) -> int:
        if map_dir is None:
            raise TopoGraphError("ObjectGoal 需要 map_dir 才能读 objects.json")
        path = map_dir / OBJECTS_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TopoGraphError(f"读 {OBJECTS_FILENAME} 失败: {path}: {exc}") from exc
        for obj in data.get("objects", []):
            if obj.get("id") == self.instance_id:
                best_crop = obj.get("best_crop")
                if not best_crop:
                    raise TopoGraphError(f"实例 {self.instance_id} 没有 best_crop")
                return int(best_crop[0])
        raise TopoGraphError(f"地图里没有实例 {self.instance_id}")


class ImageGoal:
    """按检索结果或图像描述子选目标关键帧,两种构造方式二选一。

    - ``ImageGoal(ranked_kf_ids=[...])``:检索服务已经按相似度排好名,取第一个在图里的。
    - ``ImageGoal(query_desc=..., map_descs=...)``:两边各自 L2 归一化后算余弦相似度,取最大
      那一行的行号(行号即关键帧编号,和 ``poses.txt`` 的顺序一致)。
    """

    def __init__(
        self,
        ranked_kf_ids: Optional[List[int]] = None,
        query_desc: Optional[np.ndarray] = None,
        map_descs: Optional[np.ndarray] = None,
    ) -> None:
        has_ranked = ranked_kf_ids is not None
        has_desc = query_desc is not None or map_descs is not None
        if has_ranked == has_desc:
            raise TopoGraphError(
                "ImageGoal 要给 ranked_kf_ids,或者同时给 query_desc + map_descs,二选一"
            )
        if has_desc and (query_desc is None or map_descs is None):
            raise TopoGraphError("query_desc 和 map_descs 必须同时给")
        self.ranked_kf_ids = ranked_kf_ids
        self.query_desc = query_desc
        self.map_descs = map_descs

    def resolve(self, graph: TopoGraph, map_dir: Optional[Path]) -> int:
        if self.ranked_kf_ids is not None:
            for kf in self.ranked_kf_ids:
                if kf in graph.poses:
                    return kf
            raise TopoGraphError("检索结果里没有一个关键帧在图里")
        query = self.query_desc / np.linalg.norm(self.query_desc)
        norms = np.linalg.norm(self.map_descs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self.map_descs / norms
        scores = normalized @ query
        return int(np.argmax(scores))


class TextResolver(Protocol):
    """把一句话换成一个具体 ``Goal``,由调用方注入(比如大模型或房间表)。"""

    def resolve_text(self, text: str) -> Goal:
        ...


class LanguageGoal:
    """一句话目标:实际解析交给调用方注入的 ``resolver``,本包不实现任何文字规则。"""

    def __init__(self, text: str, resolver: TextResolver) -> None:
        self.text = text
        self.resolver = resolver

    def resolve(self, graph: TopoGraph, map_dir: Optional[Path]) -> int:
        return self.resolver.resolve_text(self.text).resolve(graph, map_dir)
