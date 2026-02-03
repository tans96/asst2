import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import glm
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []
        root_bound = BoundingBox3D()
        bounding_boxes = [prim.bounding_box for prim in primitives]
        for bounding_box in bounding_boxes:
            root_bound = BoundingBox3D.union(root_bound, bounding_box)

        root_node = BVHNode(
            bound=root_bound,
            prim_left=0,
            prim_right=len(primitives),
            depth=0,
        )
        self.nodes.append(root_node)
        while len(self.nodes) < max_nodes:
            # Find the next node to split
            node_to_split_idx = -1
            max_surface_area = 0.0
            for idx, node in enumerate(self.nodes):
                if not node.is_leaf or node.prim_right - node.prim_left <= min_prim_per_node:
                    continue
                if node.bound.area > max_surface_area:
                    max_surface_area = node.bound.area
                    node_to_split_idx = idx
            if node_to_split_idx == -1:
                break  # No more nodes can be split

            node_to_split = self.nodes[node_to_split_idx]
            left_node, right_node = self.split_node(
                node_to_split,
                primitives,
                bounding_boxes,
                num_thresholds,
            )
            node_to_split.left = len(self.nodes)
            node_to_split.right = len(self.nodes) + 1
            self.nodes.append(left_node)
            self.nodes.append(right_node)
            if on_progress:
                on_progress(len(self.nodes), max_nodes)
    
    def split_node(
        self,
        node: BVHNode,
        primitives: List[Primitive],
        bounding_boxes: List[BoundingBox3D],
        num_thresholds: int,
    ) -> Tuple[BVHNode, BVHNode]:
        """
        Splits the given BVH node into two child nodes using the Surface Area Heuristic (SAH).
        The function should determine the best splitting axis and threshold to minimize
        the expected cost of traversing the BVH during ray intersection tests.

        :param node: The BVH node to split.
        :param primitives: The list of primitives in the BVH.
        :param bounding_boxes: The list of bounding boxes corresponding to the primitives.
        :param min_prim_per_node: The minimum number of primitives per leaf node.
        :param num_thresholds: The number of thresholds per axis to consider when splitting.
        :return: A tuple containing the indices of the left and right child nodes.
        """
        optimal_bucket = []
        optimal_SAH = float("inf")
        for axis in range(3):
            primitives_indices = [[] for _ in range(num_thresholds)]
            buckets = [BoundingBox3D() for _ in range(num_thresholds)]
            for i in range(node.prim_left, node.prim_right):
                center = bounding_boxes[i].center
                axis_value = int((center[axis] - node.bound.min[axis]) / (node.bound.max[axis] - node.bound.min[axis]) * num_thresholds)
                axis_value = min(axis_value, num_thresholds - 1)
                # Determine which bucket this primitive belongs to
                primitives_indices[axis_value].append(i)
                buckets[axis_value] = BoundingBox3D.union(buckets[axis_value], bounding_boxes[i])
            
            for i in range(1, num_thresholds):
                left_bboxes = BoundingBox3D()
                right_bboxes = BoundingBox3D()
                left_bucket = []
                right_bucket = []
                for j in range(i):
                    if primitives_indices[j]:
                        left_bucket.extend(primitives_indices[j])
                        left_bboxes = BoundingBox3D.union(left_bboxes, buckets[j])

                if len(left_bucket) == 0:
                    continue
                
                for j in range(i, num_thresholds):
                    if primitives_indices[j]:
                        right_bucket.extend(primitives_indices[j])
                        right_bboxes = BoundingBox3D.union(right_bboxes, buckets[j])

                if len(right_bucket) == 0:
                    continue
                
                sah = left_bboxes.area * len(left_bucket) + right_bboxes.area * len(right_bucket)
                if sah < optimal_SAH:
                    optimal_SAH = sah
                    optimal_bucket = [left_bboxes, right_bboxes, left_bucket, right_bucket] # Store axis, threshold, and buckets
        
        if not optimal_bucket:
            raise ValueError("No valid split found for the node. Something is wrong. Could be overlapping primitives?")

        left_bboxes, right_bboxes, left_bucket, right_bucket = optimal_bucket
        # Reorder primitives in-place based on the optimal split
        new_sub_array = left_bucket + right_bucket
        # Debug: Verify all primitives are accounted for
        assert len(new_sub_array) == (node.prim_right - node.prim_left), \
        f"Lost primitives! Expected {node.prim_right - node.prim_left}, got {len(new_sub_array)}"

        primitives[node.prim_left:node.prim_right] = [primitives[i] for i in new_sub_array]
        bounding_boxes[node.prim_left:node.prim_right] = [bounding_boxes[i] for i in new_sub_array]
        
        left_node = BVHNode(
            bound=left_bboxes,
            prim_left=node.prim_left,
            prim_right=node.prim_left + len(left_bucket),
            depth=node.depth + 1
        )
        right_node = BVHNode(
            bound=right_bboxes,
            prim_left=node.prim_left + len(left_bucket),
            prim_right=node.prim_right,
            depth=node.depth + 1
        )

        return left_node, right_node

def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
