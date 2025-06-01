import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R  # Radius
        self.w = w  # Width
        self.n = n  # Resolution
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.X, self.Y, self.Z = self.compute_mesh()

    def compute_mesh(self):
        """Compute the 3D mesh/grid of (x, y, z) points on the surface."""
        U, V = np.meshgrid(self.u, self.v)
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def surface_area(self):
        """Approximate the surface area using numerical integration."""
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]
        dA = np.zeros((self.n - 1, self.n - 1))
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                vec_u = self.partial_derivative_u(i, j)
                vec_v = self.partial_derivative_v(i, j)
                dA[i, j] = np.linalg.norm(np.cross(vec_u, vec_v)) * du * dv
        return np.sum(dA)

    def partial_derivative_u(self, i, j):
        """Compute the partial derivative with respect to u."""
        u_val = self.u[i]
        v_val = self.v[j]
        x_u = - (self.R + v_val * np.cos(u_val / 2)) * np.sin(u_val) - 0.5 * v_val * np.sin(u_val / 2) * np.cos(u_val)
        y_u = (self.R + v_val * np.cos(u_val / 2)) * np.cos(u_val) - 0.5 * v_val * np.sin(u_val / 2) * np.sin(u_val)
        z_u = 0.5 * v_val * np.cos(u_val / 2)
        return np.array([x_u, y_u, z_u])

    def partial_derivative_v(self, i, j):
        """Compute the partial derivative with respect to v."""
        u_val = self.u[i]
        x_v = np.cos(u_val) * np.cos(u_val / 2)
        y_v = np.sin(u_val) * np.cos(u_val / 2)
        z_v = np.sin(u_val / 2)
        return np.array([x_v, y_v, z_v])

    def edge_length(self):
        """Compute the length of the edge of the Möbius strip."""
        edge_points = self.compute_edge_points()
        length = np.sum(np.linalg.norm(np.diff(edge_points, axis=0), axis=1))
        return length

    def compute_edge_points(self):
        """Compute the points along one edge of the Möbius strip."""
        edge_u = np.linspace(0, 2 * np.pi, self.n)
        edge_v = -self.w / 2
        edge_X = (self.R + edge_v * np.cos(edge_u / 2)) * np.cos(edge_u)
        edge_Y = (self.R + edge_v * np.cos(edge_u / 2)) * np.sin(edge_u)
        edge_Z = edge_v * np.sin(edge_u / 2)
        return np.vstack((edge_X, edge_Y, edge_Z)).T

    def plot(self):
        """Visualize the Möbius strip."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title('Möbius Strip')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    R = 1.0     # Radius
    w = 0.2     # Width
    n = 100     # Resolution

    mobius = MobiusStrip(R, w, n)
    
    print(f"Surface Area: {mobius.surface_area():.4f}")
    print(f"Edge Length: {mobius.edge_length():.4f}")
    
    mobius.plot()
