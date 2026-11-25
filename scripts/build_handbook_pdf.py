"""Generates the built-in multi-chapter Python & Data Structures PDF handbook."""

from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "handbooks"
    / "Python_DSA_Reference_Guide.pdf"
)


CHAPTERS = [
    {
        "title": "Chapter 1: CPython Execution Model and Memory Management",
        "sections": [
            (
                "1.1 Bytecode Compilation and the Python Virtual Machine",
                [
                    "When CPython executes a source file (.py), it does not interpret raw characters line by line. Instead, the source code passes through lexing and parsing into an Abstract Syntax Tree (AST), which the compiler lowers into platform-independent bytecode instructions stored in .pyc files inside __pycache__. Each bytecode instruction operates on a stack-based virtual machine known as the CPython Evaluation Loop (_PyEval_EvalFrameDefault).",
                    "Because CPython is a stack machine rather than a register machine, instructions such as LOAD_FAST push object references onto an evaluation stack, BINARY_OP pops operands and pushes the result, and RETURN_VALUE unwinds the current frame. Understanding this indirection explains why tight numerical loops in pure Python incur per-iteration boxing and dispatch overhead compared to vectorized C extensions.",
                ],
                """import dis

def hypotenuse_sq(a: int, b: int) -> int:
    return a * a + b * b

# Inspect compiled stack instructions
dis.dis(hypotenuse_sq)""",
                None,
            ),
            (
                "1.2 Reference Counting and Generational Garbage Collection",
                [
                    "Every Python object in CPython carries a PyObject header containing two core fields: ob_refcnt (the active reference count) and ob_type (a pointer to the object's type descriptor). Whenever a variable, container slot, or function frame binds to an object, ob_refcnt increments; when a reference goes out of scope or is deleted with del, ob_refcnt decrements. As soon as ob_refcnt drops to zero, memory is reclaimed deterministically and immediately.",
                    "Reference counting alone cannot reclaim reference cycles—for example, two doubly-linked list nodes or parent-child tree nodes referencing each other after external pointers disappear. To solve cyclic leaks, CPython pairs reference counting with a Generational Cyclic Garbage Collector (the gc module) organized into three generations (Gen 0, Gen 1, Gen 2). Newly allocated container objects start in Generation 0; objects surviving a collection sweep are promoted to older generations that are scanned less frequently, following the weak generational hypothesis.",
                ],
                None,
                [
                    ["Mechanism", "Trigger Condition", "Strengths", "Limitation"],
                    ["Reference Counting", "ob_refcnt == 0 immediately", "Deterministic cleanup, low pause", "Cannot free reference cycles"],
                    ["Cyclic GC (Gen 0-2)", "Allocation minus deallocation threshold", "Detects unreachable object islands", "Periodic stop-the-world scan"],
                ],
            ),
            (
                "1.3 Mutability, Object Identity, and Parameter Passing",
                [
                    "Python passes function arguments by object reference (often called call-by-sharing). Rebinding a parameter name inside a function only redirects the local variable; mutating a mutable object (list, dict, set, bytearray) through that parameter modifies the caller's underlying object in place. Immutable objects (int, float, str, tuple, frozenset) allocate new instances upon transformation, making them hashable and safe as dictionary keys provided any nested tuple elements are also hashable.",
                    "A classic pitfall in Python is using a mutable default argument such as def append_item(val, bucket=[]). Default argument expressions are evaluated exactly once when the def statement executes, not on each function call. Always default to None and initialize a fresh list inside the function body.",
                ],
                """def add_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    if tags is None:
        tags = []
    tags.append(tag)
    return tags""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 2: Asymptotic Complexity and Amortized Analysis",
        "sections": [
            (
                "2.1 Big-O, Big-Omega, and Space-Time Trade-offs",
                [
                    "Asymptotic notation characterizes how runtime or auxiliary memory scales as input size n approaches infinity, abstracting away hardware constants. Big-O (O) bounds growth from above (worst-case ceiling), Big-Omega bounds growth from below, and Big-Theta denotes a tight asymptotic bound where upper and lower rates coincide.",
                    "When evaluating recursive algorithms, auxiliary call-stack depth counts toward space complexity. For instance, recursive DFS on a skewed binary tree of n nodes consumes O(n) stack frames, whereas a balanced tree requires O(log n) auxiliary stack space.",
                ],
                None,
                [
                    ["Complexity Class", "Name", "Canonical Example", "n = 10^5 Feasibility"],
                    ["O(1)", "Constant", "Hash table lookup, array index", "Instant (< 1 us)"],
                    ["O(log n)", "Logarithmic", "Binary search on sorted list", "Instant (~17 steps)"],
                    ["O(n)", "Linear", "Single-pass two-pointer scan", "Fast (~1-5 ms)"],
                    ["O(n log n)", "Linearithmic", "Merge Sort, Timsort, heap sort", "Practical (~25 ms)"],
                    ["O(n^2)", "Quadratic", "Nested pair comparisons", "Too slow (~10^10 ops)"],
                ],
            ),
            (
                "2.2 Amortized Analysis of Dynamic Arrays",
                [
                    "A Python list is implemented in C as a contiguous array of PyObject* pointers with geometric over-allocation. Appending an element via list.append(x) writes into spare capacity in O(1) time until the buffer fills. Upon overflow, CPython allocates a larger buffer (growing roughly by ~12.5% plus a small constant) and copies existing pointers.",
                    "Although an individual resize costs O(n) pointer copies, resizes occur exponentially farther apart. Summing geometric copy costs across n consecutive appends yields a total work bounded by c * n operations, proving that list.append runs in O(1) amortized time. By contrast, list.insert(0, x) or list.pop(0) shifts every subsequent pointer left or right, costing strict O(n) time per call.",
                ],
                None,
                None,
            ),
        ],
    },
    {
        "title": "Chapter 3: Hash Tables, Dictionaries, and Specialized Collections",
        "sections": [
            (
                "3.1 Compact Open-Addressing Hash Tables in dict",
                [
                    "Since Python 3.6, built-in dict uses a compact two-array open-addressing layout designed by Raymond Hettinger. Instead of storing sparse 24-byte entries directly inside the hash table slots, CPython maintains (1) a compact sparse index array storing small 1-to-8 byte signed integer indices, and (2) a dense entries array appended in insertion order storing (hash, key_ptr, value_ptr).",
                    "This split design cuts dictionary memory footprint by 20% to 25%, improves CPU cache locality during iteration, and guarantees insertion-order preservation as a language specification feature (Python 3.7+). Hash collisions are resolved via perturbed pseudo-random probing over the sparse index table rather than separate linked-list chaining. To keep probe chains short, CPython resizes the index table whenever load factor exceeds two-thirds (2/3).",
                ],
                None,
                [
                    ["Container", "Backing Structure", "Lookup", "Insert/Delete Head", "Insert/Delete Tail"],
                    ["list", "Dynamic pointer array", "O(1) by idx / O(n) val", "O(n)", "O(1) amortized"],
                    ["dict / set", "Open-addressing hash table", "O(1) avg, O(n) worst", "N/A", "O(1) amortized"],
                    ["collections.deque", "Doubly-linked 64-slot blocks", "O(n) middle", "O(1)", "O(1)"],
                ],
            ),
            (
                "3.2 Deques, Counters, and Default Dictionaries",
                [
                    "The standard library module collections provides high-performance container datatypes tailored to algorithmic workloads. collections.deque implements a double-ended queue using a doubly-linked list of fixed-size C pointer blocks, supporting thread-safe O(1) appendleft and popleft without shifting memory. Counter subclasses dict for frequency tallies and supports Counter.most_common(k) using heapq internally when k is small relative to unique keys.",
                ],
                """from collections import Counter, deque

def first_unique_char(s: str) -> int:
    counts = Counter(s)
    for idx, ch in enumerate(s):
        if counts[ch] == 1:
            return idx
    return -1""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 4: Array Patterns — Two Pointers, Sliding Window, and Prefix Sums",
        "sections": [
            (
                "4.1 Opposite-End Two Pointers and Variable Sliding Window",
                [
                    "Two-pointer techniques eliminate nested O(n^2) loops by exploiting monotonicity or sorted order. In opposite-end two pointers (such as Two Sum II on a sorted array or Container With Most Water), left starts at index 0 and right starts at n - 1; each step discards an entire row or column of suboptimal pairs in O(1) time.",
                    "A variable-size sliding window maintains an active range [left, right] satisfying a problem invariant (such as at most k distinct characters). The right pointer expands the window one step per iteration, while an inner while loop advances left only when the invariant breaks. Because both pointers move monotonically forward at most n times, total runtime across the entire loop is O(n), not O(n^2).",
                ],
                """def longest_unique_substring(s: str) -> int:
    last_seen: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best""",
                None,
            ),
            (
                "4.2 Prefix Sums with Hash Map Lookup",
                [
                    "Given an integer array nums that may contain negative values, sliding window monotonicity no longer holds because adding elements can decrease the sum. Instead, define prefix[i] = nums[0] + ... + nums[i-1]. Any subarray sum over [l, r] equals prefix[r+1] - prefix[l]. By recording frequencies of previously seen prefix sums in a dictionary, we count subarrays summing to target k in O(n) time and O(n) space.",
                ],
                """def subarray_sum_count(nums: list[int], target: int) -> int:
    seen_prefixes = {0: 1}
    running = 0
    matches = 0
    for x in nums:
        running += x
        matches += seen_prefixes.get(running - target, 0)
        seen_prefixes[running] = seen_prefixes.get(running, 0) + 1
    return matches""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 5: Linked Lists, Stacks, and Monotonic Structures",
        "sections": [
            (
                "5.1 Sentinel Nodes and Floyd's Cycle Detection",
                [
                    "Pointer manipulation on singly or doubly linked lists often requires special-casing empty lists or modifications to the head node. Introducing a dummy sentinel node (dummy = ListNode(0, head)) ensures every real node has a valid predecessor, simplifying insertion, filtering, and reversal logic.",
                    "Floyd's Tortoise and Hare algorithm detects cycles in O(n) time and O(1) space without storing visited node addresses in a set. A slow pointer advances one link per step while a fast pointer advances two links. If a cycle of length C exists, fast enters the cycle first and closes the distance to slow by exactly 1 step per iteration, guaranteeing a meeting inside the cycle. Resetting one pointer to head and advancing both at speed 1 locates the exact cycle entry node.",
                ],
                """class ListNode:
    def __init__(self, val: int = 0, nxt: "ListNode | None" = None):
        self.val = val
        self.next = nxt

def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    curr = head
    while curr is not None:
        nxt_node = curr.next
        curr.next = prev
        prev = curr
        curr = nxt_node
    return prev""",
                None,
            ),
            (
                "5.2 Monotonic Stack and Monotonic Deque Patterns",
                [
                    "A monotonic decreasing stack stores indices whose corresponding array values are strictly non-increasing from bottom to top. When scanning left-to-right and encountering a value nums[i] larger than the stack top, popping smaller indices immediately resolves their Next Greater Element in O(1) amortized time per element—each index is pushed once and popped at most once.",
                    "Extending a monotonic stack into a monotonic deque (using collections.deque) allows evicting expired indices from the front while maintaining sorted candidates at the back, solving Sliding Window Maximum in optimal O(n) time.",
                ],
                """def next_greater_elements(nums: list[int]) -> list[int]:
    result = [-1] * len(nums)
    stack: list[int] = []  # indices with unresolved next greater value
    for idx, val in enumerate(nums):
        while stack and nums[stack[-1]] < val:
            prev_idx = stack.pop()
            result[prev_idx] = val
        stack.append(idx)
    return result""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 6: Trees, Binary Search Trees, and Binary Heaps",
        "sections": [
            (
                "6.1 Tree Traversals and BST Invariants",
                [
                    "Depth-First Search (DFS) on a binary tree visits nodes in Preorder (root, left, right), Inorder (left, root, right), or Postorder (left, right, root). Breadth-First Search (BFS) explores level by level using a FIFO deque. On a valid Binary Search Tree (BST), every key in node.left is strictly smaller than node.val and every key in node.right is strictly greater—not merely the immediate children, which requires propagating (low, high) bounds down recursive validation calls.",
                    "An inorder traversal of a BST yields keys in strictly ascending sorted order. Balanced BST variants (AVL trees, Red-Black trees) bound height h to O(log n) via local tree rotations, preventing degeneration into an O(n) linked chain on sorted insertions.",
                ],
                """def is_valid_bst(node, low: float = float("-inf"), high: float = float("inf")) -> bool:
    if node is None:
        return True
    if not (low < node.val < high):
        return False
    return (
        is_valid_bst(node.left, low, node.val)
        and is_valid_bst(node.right, node.val, high)
    )""",
                None,
            ),
            (
                "6.2 Binary Heaps and Priority Queues with heapq",
                [
                    "Python's heapq module implements an array-backed binary min-heap directly inside a standard Python list. For zero-based index i, the left child sits at 2*i + 1, the right child at 2*i + 2, and the parent at (i - 1) // 2. Because the smallest item always resides at heap[0], peeking takes O(1) time while heappush and heappop restore the heap property in O(log n) time.",
                    "Crucially, heapq.heapify(arr) converts an arbitrary unsorted list of n items into a valid min-heap in linear O(n) time—not O(n log n)—by sifting downward from the last internal parent (n // 2 - 1) to index 0, where most nodes sit near the leaves and travel only 1 or 2 levels. To simulate a max-heap in Python, negate numeric priorities upon insertion and negation-restore upon pop.",
                ],
                """import heapq

def k_largest_elements(nums: list[int], k: int) -> list[int]:
    if k <= 0:
        return []
    min_heap = nums[:k]
    heapq.heapify(min_heap)
    for val in nums[k:]:
        if val > min_heap[0]:
            heapq.heapreplace(min_heap, val)
    return sorted(min_heap, reverse=True)""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 7: Graph Representations, Traversal, and Shortest Paths",
        "sections": [
            (
                "7.1 Adjacency Lists, BFS, and Topological Ordering",
                [
                    "Sparse graphs where |E| << |V|^2 should be represented using adjacency lists (dict[int, list[int]]), consuming O(V + E) memory and allowing neighbor iteration proportional to vertex degree. Adjacency matrices use O(V^2) space and are preferred only for dense graphs or algebraic path queries.",
                    "Unweighted shortest paths are solved in O(V + E) time using Breadth-First Search (BFS) initialized with source distance 0. Directed Acyclic Graphs (DAGs) admit a Topological Ordering via Kahn's BFS in-degree peeling algorithm: enqueue all vertices with in-degree 0, pop each vertex, decrement neighbors' in-degrees, and enqueue newly zeroed neighbors. If fewer than V vertices are output, the directed graph contains a cycle.",
                ],
                """from collections import deque

def kahn_topological_sort(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    adj = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    queue = deque(i for i in range(num_nodes) if indegree[i] == 0)
    order: list[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return order if len(order) == num_nodes else []""",
                None,
            ),
            (
                "7.2 Dijkstra's Algorithm and Disjoint Set Union (Union-Find)",
                [
                    "For weighted graphs with non-negative edge weights, Dijkstra's algorithm greedily finalizes vertices in order of tentative distance using a min-heap priority queue in O((V + E) log V) time. Lazy deletion is idiomatic in Python: rather than mutating priorities inside heapq, push updated (new_dist, node) tuples and skip stale popped entries whenever popped_dist > dist[node]. If negative edge weights exist, Dijkstra's greedy invariant fails and Bellman-Ford (O(V * E)) must be used.",
                    "Disjoint Set Union (Union-Find) tracks dynamic connected components across undirected edges. Combining Path Compression in find(x) with Union by Rank/Size reduces amortized cost per operation to O(alpha(V)), where alpha is the inverse Ackermann function (effectively <= 4 for all practical inputs).",
                ],
                """import heapq

def dijkstra_shortest_paths(adj: dict[int, list[tuple[int, int]]], start: int) -> dict[int, int]:
    dist = {start: 0}
    pq: list[tuple[int, int]] = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, weight in adj.get(u, []):
            cand = d + weight
            if cand < dist.get(v, float("inf")):
                dist[v] = cand
                heapq.heappush(pq, (cand, v))
    return dist""",
                [
                    ["Graph Algorithm", "Edge Constraint", "Time Complexity", "Primary Use Case"],
                    ["BFS", "Unweighted (w = 1)", "O(V + E)", "Minimum hops / shortest unweighted path"],
                    ["Dijkstra", "Non-negative (w >= 0)", "O((V + E) log V)", "Single-source weighted shortest path"],
                    ["Bellman-Ford", "Arbitrary (negative ok)", "O(V * E)", "Negative weights & negative cycle check"],
                    ["Union-Find (DSU)", "Undirected connectivity", "O(alpha(V)) / op", "Kruskal MST, dynamic component queries"],
                ],
            ),
        ],
    },
    {
        "title": "Chapter 8: Binary Search, Merge Sort, QuickSort, and Timsort",
        "sections": [
            (
                "8.1 Boundary-Safe Binary Search on Monotonic Predicates",
                [
                    "Binary search applies far beyond finding an exact number in a sorted list: any problem featuring a monotonic boolean predicate P(x) that transitions once from False to True over a discrete range [lo, hi] can be solved in O(log(hi - lo)) evaluations. This powers capacity optimization problems such as minimum ship capacity within D days or Koko eating bananas.",
                    "Python's bisect module provides C-accelerated bisect_left (first index >= target) and bisect_right (first index > target). Maintaining half-open or closed loop invariants consistently avoids off-by-one infinite loops.",
                ],
                """def binary_search_first_true(lo: int, hi: int, predicate) -> int:
    # Returns smallest x in [lo, hi] where predicate(x) is True, or hi + 1 if none
    ans = hi + 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if predicate(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans""",
                None,
            ),
            (
                "8.2 Divide-and-Conquer Sorting and Python's Timsort",
                [
                    "Merge Sort recursively splits an array in half, sorts both halves, and merges them stably in guaranteed O(n log n) worst-case time using O(n) auxiliary buffer space. QuickSort partitions elements around a pivot in place (O(log n) average stack space, O(n log n) expected time), though adversarial pivot selection degrades naive QuickSort to O(n^2) unless randomized pivots or median-of-three are used.",
                    "Python's built-in list.sort() and sorted() use Timsort (invented by Tim Peters in 2002), a hybrid stable sorting algorithm combining Insertion Sort and Merge Sort. Timsort scans data for existing ascending or descending 'natural runs', extends short runs to a minimum run length (minrun between 32 and 64) via binary insertion sort, and merges runs on a stack using galloping mode when one run consistently beats another. Consequently, Timsort achieves O(n) best-case time on already-sorted or nearly-sorted real-world data and O(n log n) worst-case time with O(n) worst-case temporary memory.",
                ],
                None,
                [
                    ["Algorithm", "Best Time", "Average Time", "Worst Time", "Space", "Stable?"],
                    ["Insertion Sort", "O(n)", "O(n^2)", "O(n^2)", "O(1)", "Yes"],
                    ["Merge Sort", "O(n log n)", "O(n log n)", "O(n log n)", "O(n)", "Yes"],
                    ["QuickSort (random)", "O(n log n)", "O(n log n)", "O(n^2)", "O(log n)", "No"],
                    ["Timsort (Python)", "O(n)", "O(n log n)", "O(n log n)", "O(n)", "Yes"],
                ],
            ),
        ],
    },
    {
        "title": "Chapter 9: Dynamic Programming and Memoization",
        "sections": [
            (
                "9.1 Overlapping Subproblems, Top-Down Memoization, and Tabulation",
                [
                    "Dynamic Programming (DP) optimizes recursive search when two properties hold: Optimal Substructure (an optimal global solution composes optimal solutions to smaller subproblems) and Overlapping Subproblems (naive recursion recomputes identical state tuples exponentially many times).",
                    "In Python, top-down memoization is concise via @functools.lru_cache(maxsize=None) or @functools.cache, provided all state arguments are hashable. Bottom-up tabulation evaluates states iteratively in topological dependency order, eliminating recursion-limit issues (sys.getrecursionlimit() defaults to 1000) and enabling rolling-array space optimization—for instance, reducing 2D grid or knapsack DP from O(n * W) memory down to a single 1D row of O(W).",
                ],
                """def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    dp = [0] * (capacity + 1)
    for w, val in zip(weights, values):
        # Iterate backwards so each item is used at most once
        for cap in range(capacity, w - 1, -1):
            dp[cap] = max(dp[cap], dp[cap - w] + val)
    return dp[capacity]""",
                None,
            ),
            (
                "9.2 Patience Sorting and O(n log n) Longest Increasing Subsequence",
                [
                    "Standard DP for Longest Increasing Subsequence (LIS) computes dp[i] = 1 + max(dp[j] for j < i if nums[j] < nums[i]) in O(n^2) time. Maintaining an auxiliary array tails where tails[k] stores the smallest possible ending value of an increasing subsequence of length k + 1 makes tails strictly increasing, allowing bisect_left updates in O(n log n) time.",
                ],
                """from bisect import bisect_left

def length_of_lis(nums: list[int]) -> int:
    tails: list[int] = []
    for x in nums:
        pos = bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)""",
                None,
            ),
        ],
    },
    {
        "title": "Chapter 10: Iterators, Generators, Decorators, and Context Managers",
        "sections": [
            (
                "10.1 Lazy Evaluation with Iterators and Generators",
                [
                    "An iterable implements __iter__() returning an iterator; an iterator implements __next__() returning consecutive items and raising StopIteration when exhausted. Generator functions containing yield compile into resumable frame state machines: calling a generator function does not run its body immediately, but returns a generator object that suspends execution and local variables at each yield expression.",
                    "Streaming multi-gigabyte log files or database cursors through generator pipelines keeps peak resident memory at O(1) instead of materializing full intermediate lists in RAM.",
                ],
                """from typing import Iterator

def chunked_lines(filepath: str, batch_size: int = 500) -> Iterator[list[str]]:
    batch: list[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            batch.append(line.rstrip("\\n"))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch""",
                None,
            ),
            (
                "10.2 Function Decorators and Deterministic Resource Cleanup",
                [
                    "A decorator is a higher-order callable wrapping another function to add cross-cutting behavior such as caching, retry backoff, timing telemetry, or authorization checks. Always apply @functools.wraps(fn) to the inner wrapper so __name__, __doc__, and type annotations are preserved for introspection tools.",
                    "Context managers implement __enter__() and __exit__(exc_type, exc_val, exc_tb) (or use @contextlib.contextmanager around a single yield inside a try/finally block) to guarantee deterministic cleanup of file descriptors, network sockets, database transactions, and thread locks even when exceptions unwind the call stack.",
                ],
                """import time
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"{fn.__name__} completed in {elapsed_ms:.2f} ms")
    return wrapper""",
                None,
            ),
        ],
    },
]


def _draw_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#6B5B4E"))
    canvas.drawString(0.75 * inch, 10.45 * inch, "Python & Data Structures Reference Guide — CourseRAG Knowledge Base")
    canvas.drawRightString(7.75 * inch, 0.50 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D6C8B2"))
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 10.35 * inch, 7.75 * inch, 10.35 * inch)
    canvas.line(0.75 * inch, 0.65 * inch, 7.75 * inch, 0.65 * inch)
    canvas.restoreState()


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()

    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#2C221E"),
        spaceAfter=12,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#6B5B4E"),
        spaceAfter=20,
    )
    chapter_style = ParagraphStyle(
        "ChapterHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=21,
        textColor=colors.HexColor("#8C4627"),
        spaceBefore=6,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#2C221E"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCopy",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=14.8,
        textColor=colors.HexColor("#241C18"),
        spaceAfter=8,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=8.7,
        leading=11.8,
        textColor=colors.HexColor("#2C221E"),
        backColor=colors.HexColor("#F4EFE6"),
        borderColor=colors.HexColor("#D6C8B2"),
        borderWidth=0.6,
        borderPadding=7,
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=10,
    )

    story = []

    # Title / Overview section on Page 1
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Python Internals &amp; Data Structures Reference Guide", cover_title))
    story.append(
        Paragraph(
            "Concise computer science study handbook covering CPython memory mechanics, "
            "asymptotic analysis, core data structures, graph and sorting algorithms, "
            "dynamic programming, and idiomatic Python patterns.",
            cover_sub,
        )
    )

    toc_rows = [["Chapter", "Core Topics Covered"]]
    for idx, ch in enumerate(CHAPTERS, start=1):
        clean_name = ch["title"].split(": ", 1)[1]
        sec_names = "; ".join(s[0].split(" ", 1)[1] for s in ch["sections"])
        toc_rows.append([f"Chapter {idx}: {clean_name}", sec_names])

    toc_table = Table(toc_rows, colWidths=[2.7 * inch, 4.3 * inch])
    toc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8DEC8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C221E")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FAF7F0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6C8B2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(toc_table)
    story.append(PageBreak())

    for ch_idx, chapter in enumerate(CHAPTERS):
        story.append(Paragraph(chapter["title"], chapter_style))
        for sec_title, paragraphs, code_snippet, table_data in chapter["sections"]:
            story.append(Paragraph(sec_title, section_style))
            for para in paragraphs:
                escaped = (
                    para.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(escaped, body_style))

            if table_data:
                num_cols = len(table_data[0])
                col_w = (7.0 * inch) / num_cols
                tbl = Table(table_data, colWidths=[col_w] * num_cols)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8DEC8")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C221E")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                            ("LEADING", (0, 0), (-1, -1), 11.5),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FAF7F0")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6C8B2")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(Spacer(1, 0.05 * inch))
                story.append(tbl)
                story.append(Spacer(1, 0.12 * inch))

            if code_snippet:
                story.append(Preformatted(code_snippet.strip(), code_style))

        if ch_idx < len(CHAPTERS) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.82 * inch,
        bottomMargin=0.78 * inch,
        title="Python & Data Structures Reference Guide",
        author="CourseRAG Study Series",
    )
    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return output_path


if __name__ == "__main__":
    built = build_pdf()
    print(f"Built handbook PDF: {built} ({built.stat().st_size:,} bytes)")
