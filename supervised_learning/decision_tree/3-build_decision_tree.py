#!/usr/bin/env python3
"""
Decision Tree Framework.
"""
import numpy as np

class Node:
    """
    Represents an internal decision (split) node within the decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
    def left_child_add_prefix(self, text):
        """ Add prefix to left child """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text
    def get_leaves_below(self):
        """
        Recursively collects all leaf nodes beneath this node."""
        leaves = []

        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())

        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves
    def right_child_add_prefix(self, text):
        """ Add prefix to right child """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            label = (
                f"root [feature={self.feature}, threshold={self.threshold}]"
            )
        else:
            label = (
                f"-> node [feature={self.feature}, threshold={self.threshold}]"
            )
        left_str = self.left_child_add_prefix(self.left_child.__str__())
        right_str = self.right_child_add_prefix(self.right_child.__str__())
        return label + "\n" + left_str + right_str  # remove .rstrip("\n")
    def max_depth_below(self) :
        """
        Recursively calculates the maximum absolute depth reached
        """

        if self.left_child is None and self.right_child is None:
            return self.depth

        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth

        return max(left_depth, right_depth)
    def count_nodes_below(self, only_leaves=False):
        """
        Recursively calculates the maximum absolute depth reached
        """
        # If leaf-like endpoint (no children)
        left_count = self.left_child.count_nodes_below(only_leaves) if self.left_child else 0
        right_count = self.right_child.count_nodes_below(only_leaves) if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count

class Leaf(Node):
    """
    Represents a terminal node (leaf) of the decision tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """ Recursively collects all leaf nodes beneath this node."""
        return [self]

    def max_depth_below(self) :
        """
        Returns the leaf's own depth, serving as the recursion base case.
        """

        return self.depth
    def count_nodes_below(self, only_leaves=False):
        """
        Returns the leaf's own depth, serving as the recursion base case.
        """

        return 1

class Decision_Tree():
    """
    The orchestrator class for managing the decision tree structure.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None
    def __str__(self):
        return self.root.__str__()
    def depth(self) :
        """
        Calculates the maximum depth of the entire tree from the root.
        """
        return self.root.max_depth_below()
    def get_leaves(self):
        """ Recursively collects all leaf nodes beneath the root node."""
        return self.root.get_leaves_below()
    def count_nodes(self, only_leaves=False):
        """
        Counts the total number of nodes or leaves down the tree hierarchy.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
