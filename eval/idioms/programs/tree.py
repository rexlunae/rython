"""A recursive class with Optional child fields, a generator with
yield from, and recursion depth/count."""
from typing import Iterator, Optional


class Node:
    def __init__(self, key: int):
        self.key = key
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def insert(self, key: int) -> None:
        if key < self.key:
            if self.left is None:
                self.left = Node(key)
            else:
                self.left.insert(key)
        else:
            if self.right is None:
                self.right = Node(key)
            else:
                self.right.insert(key)

    def inorder(self) -> Iterator[int]:
        if self.left is not None:
            yield from self.left.inorder()
        yield self.key
        if self.right is not None:
            yield from self.right.inorder()

    def depth(self) -> int:
        l = self.left.depth() if self.left is not None else 0
        r = self.right.depth() if self.right is not None else 0
        return 1 + max(l, r)

    def contains(self, key: int) -> bool:
        node: Optional[Node] = self
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False


def main() -> None:
    root = Node(50)
    for k in [30, 70, 20, 40, 60, 80, 35]:
        root.insert(k)
    print(list(root.inorder()))
    print("depth", root.depth())
    print([root.contains(k) for k in (35, 36, 80)])
    evens = [k for k in root.inorder() if k % 2 == 0]
    print(evens, sum(evens))
    root.left.insert(45)
    print(list(root.inorder())[:5], root.left.right.right.key)


if __name__ == "__main__":
    main()
