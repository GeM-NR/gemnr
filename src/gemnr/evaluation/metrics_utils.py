def is_connected(n, edges):
    if n == 0:
        return True  # An empty graph is trivially connected

    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)  # assuming undirected graph

    # DFS to check connectivity
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)

    # Start DFS from node 0
    dfs(0)

    # Graph is connected if all nodes were visited
    return len(visited) == n