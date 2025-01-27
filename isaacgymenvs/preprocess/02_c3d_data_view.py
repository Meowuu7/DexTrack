import os
from c3d.scripts.c3d_viewer import *


if __name__ == '__main__':

    fileroot = '/home/mingxian/project/grab/mocap__s2/s2'
    filename_ = 'camera_lift.c3d' 
    filelist = ['hammer_lift.c3d' ]

    filename = os.path.join(fileroot, filename_)
    reader = c3d.Reader(open(filename, 'rb')) 
    frames = list(reader.read_frames())
    frames_len = len(frames)

    '''
    frame_id, points, analogs = frames[i]
    pts = points[:, :3] # 前3为xyz
    reader.point_rate: 120.0
    reader.point_labels # 每个点对应的名称
    '''
    

    debug = 10


    
    exit
    # Viewer(c3d.Reader(open(filename, 'rb'))).mainloop()
    for filename in filelist:
        filename = os.path.join(fileroot, filename)
        try:
            viewer = Viewer(c3d.Reader(open(filename, 'rb')))

            # viewer.on_key_press()
            viewer.mainloop()
        except StopIteration:
            pass

    pass