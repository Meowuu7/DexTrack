import taichi as ti

import numpy as np


# import polyscope as ps

# ps.init()
# ps.set_ground_plane_mode("none")
# ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
# ps.set_screenshot_extension(".png")


color = [
    (0, 191 / 255.0, 255 / 255.0),
    (186 / 255.0, 85 / 255.0, 211 / 255.0),
    (255 / 255.0, 81 / 255.0, 81 / 255.0),
    (92 / 255.0, 122 / 255.0, 234 / 255.0),
    (255 / 255.0, 138 / 255.0, 174 / 255.0),
    (77 / 255.0, 150 / 255.0, 255 / 255.0),
    (192 / 255.0, 237 / 255.0, 166 / 255.0)
    #
]


def get_2D_sdfs(sdf_fn):
    sdf_values = np.load(sdf_fn)
    print(sdf_values.shape)
    
    sv_sdf_folder = "softzoo/assets"
    sv_sdf_values = sdf_values[30, :, :]
    
    ## a numpy array in 2D domain ##
    ori_sdf_fn = sdf_fn.split("/")[-1].split(".")[0]
    sv_sdf_fn = f"{sv_sdf_folder}/{ori_sdf_fn}.npy"
    np.save(sv_sdf_fn, sv_sdf_values)
    print(f"Saved 2D sdf values to {sv_sdf_fn}")
    
    ### get pts from the sdf values ##
    tot_pts = []
    nn_x = sv_sdf_values.shape[0]
    nn_y = sv_sdf_values.shape[1]
    for i_x in range(nn_x):
        for i_y in range(nn_y):
            cur_pos_xy = [float(i_x) / float(nn_x), float(i_y) / float(nn_y)]
            if sv_sdf_values[i_x, i_y] <= 0.0:
                tot_pts.append(cur_pos_xy)
    tot_pts = np.array(tot_pts, dtype=np.float32)
    print(f"Total points: {tot_pts.shape}")
    
    sv_pts_fn = f"{sv_sdf_folder}/{ori_sdf_fn}_pts.npy"
    np.save(sv_pts_fn, tot_pts)
    print(f"pts saved to {sv_pts_fn }")
    
    # ps.register_point_cloud("tot_pts", tot_pts, radius=0.05, color=color[0])
    # # ps.show()
    # # ps.scre
    # ps.remove_all_point_clouds()
    
    

# python softzoo/tools/get_2D_sdf_grids.py

if __name__=='__main__':
    sdf_fn = "/data/xueyi/GRAB/GRAB_extracted_test/train/102_obj.npy"
    get_2D_sdfs(sdf_fn)

