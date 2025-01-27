import os
from c3d.scripts.c3d_viewer import *


if __name__ == '__main__':

    fileroot = '/home/mingxian/project/grab/mocap__s2/s2'
    filename = 'hammer_lift.c3d' 
    filelist = ['hammer_lift.c3d' ]

    # for filename in filelist:
    #     filename = os.path.join(fileroot, filename)
    #     try:
    #         Viewer(c3d.Reader(open(filename, 'rb'))).mainloop()
    #     except StopIteration:
    #         pass

    for filename in filelist:
        filename = os.path.join(fileroot, filename)
        try:
            viewer = Viewer(c3d.Reader(open(filename, 'rb')))

            # viewer.on_key_press()
            viewer.mainloop()
        except StopIteration:
            pass

    pass