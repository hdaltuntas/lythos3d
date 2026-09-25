import numpy as np

class Node:
    """Represents a 3D node in the finite element mesh."""
    def __init__(self, node_id, x, y, z):
        self.id = node_id
        self.coords = np.array([x, y, z], dtype=float)
        # Assuming 3 degrees of freedom (u_x, u_y, u_z) per node for standard 3D solid elements
        self.dofs = []

    def __repr__(self):
        return f"Node(id={self.id}, coords={self.coords})"


class Element3D:
    """Represents a general 3D finite element."""
    def __init__(self, element_id, nodes, material=None):
        self.id = element_id
        self.nodes = nodes  # List of Node objects
        self.material = material

    def get_coordinates(self):
        return np.array([node.coords for node in self.nodes])

    def get_dofs(self):
        dofs = []
        for node in self.nodes:
            dofs.extend(node.dofs)
        return dofs

    def __repr__(self):
        node_ids = [node.id for node in self.nodes]
        return f"Element3D(id={self.id}, nodes={node_ids})"

class Mesh:
    """Represents the finite element mesh."""
    def __init__(self):
        self.nodes = {}
        self.elements = {}

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_element(self, element):
        self.elements[element.id] = element
