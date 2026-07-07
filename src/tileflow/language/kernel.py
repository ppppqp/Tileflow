from typing import Any


class Kernel:
    def __init__(self, *grid: Any, threads: Any = None, cluster_dims: Any = None):
        self.grid = grid
        self.threads = threads
        self.cluster_dims = cluster_dims

    def __enter__(self):
        names = [f"b{i}" for i in range(len(self.grid))]
        if len(names) == 1:
            return names[0]
        return tuple(names)

    def __exit__(self, exc_type, exc, tb):
        return False
