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
        self.lower = {}  # lower bounds per feature
        self.upper = {}  # upper bounds per feature
    def left_child_add_prefix(self, text):
        """ Add prefix to left child """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text
    def update_bounds_below(self):
        """ Update bounds below function """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            # Copy parent bounds
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            # Update bounds for the splitting feature
            if self.feature is not None:
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        # Recurse
        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()
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
    def update_indicator(self):
        """ update indicator """

        def is_large_enough(x):
            """ check if feature satisfies lower bound """
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower.keys()
                ]),
                axis=0
            )
    def update_indicator(self):
        """ update indicator """

        def is_large_enough(x):
            """ check if feature satisfies lower bound """
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower.keys()
                ]),
                axis=0
            )

        def is_small_enough(x):
            """ check if feature satisfies upper bound """
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([
                    np.less_equal(x[:, key], self.upper[key])
                    for key in self.upper.keys()
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """ Predict node """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)
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
    def update_bounds_below(self) :
        """Pass"""
        pass 
    def pred(self, x):
        """ Predict leaf """
        return self.value

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
    def update_bounds(self):
        """
        Updates the bounds of the entire tree from the root."""
        self.root.update_bounds_below()
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
    def update_predict(self):
        """ Faster predict """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([
                leaf.indicator(A) * leaf.value
                for leaf in leaves
            ]),
            axis=0
        )