import numpy as np
import pyrender
import trimesh

from pycpd import RigidRegistration, AffineRegistration
def scale_mesh(mesh, factor=1):

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    vertices *= factor 

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)




def rigid_align_meshes(a_mesh, m_mesh):

    anchor_vertices = np.asarray(a_mesh.vertices)
    moving_vertices = np.asarray(m_mesh.vertices)

    reg = RigidRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()
    # save both the aligned mesh and also the rigid transform
    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=m_mesh.faces)
    # save the sacle, translation, and rotation and anchor mesh name to a json file
    scale = reg.s
    R = reg.R.tolist()
    t = reg.t.tolist()
    trans_dict = {
            's': scale,
            'R': R,
            't': t,
            }

    return new_mesh, trans_dict



def affine_align_meshes(a_mesh, m_mesh):

    anchor_vertices = np.asarray(a_mesh.vertices)
    moving_vertices = np.asarray(m_mesh.vertices)

    reg = AffineRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()
    # save both the aligned mesh and also the rigid transform
    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=m_mesh.faces)
    # save the sacle, translation, and rotation and anchor mesh name to a json file
    B = reg.B
    t = reg.t.tolist()
    trans_dict = {
            'B': B,
            't': t,
            }

    return new_mesh, trans_dict



def render_genereted_sdf(sdf_file):

    points = np.load(sdf_file)
    pos_points = points['pos_points']
    pos_sdf = points['pos_sdf']
    neg_points = points['neg_points']
    neg_sdf = points['neg_sdf']
    pos_points = pos_points[pos_sdf < .2, :][::20]
    neg_points = neg_points[neg_sdf > -.2, :][::20]

    colors_pos = np.zeros(pos_points.shape)
    colors_neg = np.zeros(neg_points.shape)
    colors_pos[:,0] = 1
    colors_neg[:,2] = 1
    colors = np.concatenate([colors_pos, colors_neg])

    points = np.concatenate([pos_points, neg_points])

    points[:,2] *= 5/2
    # temp = np.copy(points[:,0])
    # points[:,0] = np.copy(points[:,1])
    # points[:,1] = temp
    points[:,:2] *= -1

    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
