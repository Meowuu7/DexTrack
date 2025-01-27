import argparse
import numpy as np
import trimesh
import mesh_to_sdf
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pynput.keyboard import Controller as Keyboard
from pynput.mouse import Controller as Mouse
from pynput.mouse import Button

from .general_utils import extract_part_pca


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-path', type=str, default=None)
    parser.add_argument('--pcd-path', type=str, default=None)
    parser.add_argument('--use-surface-point-cloud', action='store_true', default=False)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--num-points', type=int, default=50000)
    parser.add_argument('--pca-components', nargs='+', type=int, default=[])
    parser.add_argument('--run', nargs='+', type=str, default=['mark_group', 'mark_passive', 'edit_pc', 'run_clustering'],
                        choices=['mark_group', 'mark_passive', 'None', 'edit_pc', 'run_clustering'])
    parser.add_argument('--out-path', type=str, default=None)
    args = parser.parse_args()

    if 'None' in args.run:
        args.run = []

    # Read mesh and convert to solid point cloud
    if args.mesh_path is not None:
        mesh = trimesh.load(args.mesh_path)

        if args.use_surface_point_cloud:
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=args.num_points)
        else:
            pcd = mesh_to_solid_point_cloud(mesh, num_points=args.num_points, to_pcd=True)
    elif args.pcd_path is not None:
        pcd = o3d.io.read_point_cloud(args.pcd_path)
    else:
        raise ValueError(f'Need to specify either {args.mesh_path} or {args.pcd_path}')
    pcd_extents = pcd.get_max_bound() - pcd.get_min_bound()

    # Run edit
    if 'edit_pc' in args.run:
        o3d.visualization.draw_geometries_with_editing([pcd])

    # Run clustering
    if 'run_clustering' in args.run:
        points = np.asarray(pcd.points)
        points_std = StandardScaler().fit_transform(points)

        pca = PCA(n_components=3)
        pca.fit(points_std)
        feats = pca.transform(points_std)

        if len(args.pca_components) > 0:
            feats_centered = feats - feats.mean(0)[None,:] # already centered by standardizer but just in case
            hyperplane_sdf = []
            for comp_i in args.pca_components:
                vec = pca.components_[comp_i, :]
                hp_sdf = (feats_centered * vec[None,:]).sum(1)
                hyperplane_sdf.append(hp_sdf)
            hyperplane_sdf = np.stack(hyperplane_sdf, axis=-1)
            feats = np.concatenate([feats, hyperplane_sdf], axis=1)
        
        kmeans = KMeans(n_clusters=args.n_clusters, init='k-means++', random_state=42)
        kmeans.fit(feats)
        labels = kmeans.labels_

        max_label = labels.max()
        print(f'point cloud has {max_label + 1} clusters')
        assert max_label < 60
        disc_colors = np.array(plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors)
        colors = disc_colors[labels]
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    if 'mark_group' in args.run:    
        # Editing
        vis = o3d.visualization.VisualizerWithVertexSelection()
        print('Press `Shift + left click` to select point')
        print('Press `1` to render color')
        print('Once two points are selected, we copy the color of the first point to all points with the same color of the second point.')
        keyboard = Keyboard()
        mouse = Mouse()
        def selection_changed_callback():
            picked_points = vis.get_picked_points() # queue structure
            if len(picked_points) > 1:
                tgt_index = picked_points[-1].index
                src_index = picked_points[0].index
                tgt_color = pcd.colors[tgt_index].copy()
                src_color = pcd.colors[src_index].copy()
                print(f'#{src_index} {src_color} ==> #{tgt_index} {tgt_color}')
                colors = np.asarray(pcd.colors)
                src_mask = np.all(colors == src_color, axis=1)
                colors[src_mask, :] = tgt_color # in-place updating pcd.color
                vis.clear_picked_points()

                keyboard.tap('1') # NOTE: not working not sure why

        vis.create_window()
        vis.add_geometry(pcd)
        vis.register_selection_changed_callback(selection_changed_callback)
        vis.run()
        vis.destroy_window()

    if 'mark_passive' in args.run:
        vis = o3d.visualization.VisualizerWithVertexSelection()
        passive_color = np.array([0, 0, 0])
        def selection_changed_callback():
            picked_points = vis.get_picked_points() # queue structure
            if len(picked_points) > 1:
                idx1 = picked_points[-1].index
                idx2 = picked_points[0].index
                color1 = pcd.colors[idx1].copy()
                color2 = pcd.colors[idx2].copy()
                if np.all(color1 == color2):
                    colors = np.asarray(pcd.colors)
                    mask = np.all(colors == color1, axis=1)
                    colors[mask, :] = passive_color # in-place update
                vis.clear_picked_points()

        vis.create_window()
        vis.add_geometry(pcd)
        vis.register_selection_changed_callback(selection_changed_callback)
        vis.run()
        vis.destroy_window()

    # Visualize principle direction
    all_part_pca_components, all_part_pca_singular_values, all_part_pc = extract_part_pca(pcd)
    
    use_line_set = False
    if use_line_set:
        line_set_points = []
        line_set_lines = []
        line_set_colors = []
    else:
        line_meshes = []
    for i, (k, part_pca_components) in enumerate(all_part_pca_components.items()):
        start = all_part_pc[k].mean(0)
        end1 = start + part_pca_components[0] * all_part_pc[k].std(0)[0]
        end2 = start + part_pca_components[1] * all_part_pc[k].std(0)[1]
        end3 = start + part_pca_components[2] * all_part_pc[k].std(0)[2]
        line_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],]
        if use_line_set:
            line_set_points.extend([start, end1, end2, end3])
            base = i * 4
            line_set_lines.extend([
                [base, base + 1],
                [base, base + 2],
                [base, base + 3],
            ])
            line_set_colors.extend([line_colors])
        else:
            for end_i, end in enumerate([end1, end2, end3]):
                line_mesh = get_arrow(start, end)
                line_mesh.paint_uniform_color(line_colors[end_i])
                line_meshes.append(line_mesh)
    if use_line_set:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(line_set_lines)
        line_set.colors = o3d.utility.Vector3dVector(line_set_colors)

        o3d.visualization.draw_geometries([pcd, line_set])
    else:
        o3d.visualization.draw_geometries([pcd] + line_meshes)

    # Save point cloud
    if args.out_path is not None:
        print(f'Save new point cloud to {args.out_path}')
        o3d.io.write_point_cloud(args.out_path, pcd)


def mesh_to_solid_point_cloud(mesh, num_points=50000, to_pcd=False):
    extents = mesh.bounds[1] - mesh.bounds[0]
    query_points = np.random.uniform(0., 1., (num_points, 3)) * extents + mesh.bounds[0]
    sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                  query_points, surface_point_method='scan',
                                  sign_method='depth', # NOTE: need to use depth otherwise lead to noise in point cloud
                                  bounding_radius=None,
                                  scan_count=100,
                                  scan_resolution=400,
                                  sample_point_count=10000000,
                                  normal_sample_count=11)
    points = query_points[sdf < 0]

    if to_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        out = pcd
    else:
        out = points

    return out


def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)


def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = min(1., scale/10)
    cylinder_radius = min(0.5, scale/20)
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)


def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)


if __name__ == '__main__':
    main()
