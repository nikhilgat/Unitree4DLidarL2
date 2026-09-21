"""Planar-patch segmentation by normal-consistent region growing on a voxelised cloud."""
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def voxelize(xyz, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    return pcd.voxel_down_sample(voxel)


def estimate_normals(pcd, knn=24):
    """Normals oriented toward the sensor at the origin."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return np.asarray(pcd.normals)


def planar_segments(pts, normals, radius=0.03, max_angle_deg=10.0, max_plane_dist=0.015):
    """Label points so that each label is a smooth, locally planar surface.

    Two neighbours (within `radius`) are linked when their normals differ by less than
    `max_angle_deg` and each lies within `max_plane_dist` of the other's tangent plane.
    Returns integer labels (one connected component per surface).
    """
    tree = cKDTree(pts)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    cos_ok = np.abs(np.einsum("ij,ij->i", normals[i], normals[j])) > np.cos(np.radians(max_angle_deg))
    d = pts[j] - pts[i]
    d_ok = (np.abs(np.einsum("ij,ij->i", d, normals[i])) < max_plane_dist) & \
           (np.abs(np.einsum("ij,ij->i", d, normals[j])) < max_plane_dist)
    keep = cos_ok & d_ok
    i, j = i[keep], j[keep]
    n = len(pts)
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels
