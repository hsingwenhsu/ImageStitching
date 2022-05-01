#from testcode.utils import show_img
import cv2
import numpy as np
import utils
import sys
data_path = '../data/'
if __name__=='__main__':
    meta_file = sys.argv[1]
    output = sys.argv[2]
    path, focal, namelist = utils.read_data(data_path+meta_file)
    print(path)
    print(focal)
    print(namelist)
    
    img_num = len(namelist)
    xoffsets = [0]*img_num # xoffset for each image
    yoffsets = [0]*img_num
    blend_left = [0]*img_num # pairs of left and right bound on the lhs
    blend_right = [0]*img_num
    des_vecs = []
   
    dims = []
    warped = []
    # get all the warped images
    for i in range(img_num):
        img = cv2.imread(path+namelist[i])
        warped_tmp = utils.get_warped(img, focal)
        h, w = warped_tmp.shape[:2]
        warped.append(warped_tmp)
        dims.append((h, w))

    # get descriptors 
    for i in range(img_num):
        h, w = warped[i].shape[:2]
        fpt = utils.get_feature_pts(warped[i], k=0.04)
        dleft, dright = utils.get_descriptors(warped[i], fpt)
        des_vecs.append((dleft, dright))

    # matching
    for i in range(img_num-1):
        h1, w1 = dims[i][:2]
        h2, w2 = dims[i+1][:2]

        des_right = des_vecs[i][1]
        des_left = des_vecs[i+1][0]
        matched_pt = utils.get_matched_pts(des_right, des_left)
        final_matched = utils.ransac(matched_pt)

        xoffset, yoffset = utils.get_stitch_info(final_matched, xoffsets[i], yoffsets[i])
        xoffsets[i+1] = xoffset
        yoffsets[i+1] = yoffset
        overlap = xoffsets[i]+w1-xoffsets[i+1]
        blend_right[i] = overlap
        blend_left[i+1] = overlap

        # for debuging
        best_x1, best_y1, best_x2, best_y2 = final_matched[0][:4]
        debug = np.zeros((max(h1, h2), w1+w2, 3), dtype='float32')
        debug[:h1, :w1] = warped[i]
        debug[:h2, w1:w1+w2] = warped[i+1]

    # information for stitching
    ymin = abs(min(yoffsets))
    yoffsets = [item+ymin for item in yoffsets]
    yheights = []
    xwidths = []
    for i in range(len(yoffsets)):
        yheights.append(dims[i][0]+yoffsets[i])
        xwidths.append(dims[i][1]+xoffsets[i])
    ch = max(yheights)
    cw = max(xwidths)
    canvas = np.zeros((ch, cw, 3), dtype='float32')
   
    # start stitching
    for i in range(img_num):
        tmp = warped[i].copy()
        h1, w1 = tmp.shape[:2]
        if blend_right[i]!=0:
            weight1 = np.linspace(1, 0, blend_right[i])
            weight1 = np.tile(weight1, (h1, 1))
            tmp[:, w1-blend_right[i]:, 0]*=weight1
            tmp[:, w1-blend_right[i]:, 1]*=weight1
            tmp[:, w1-blend_right[i]:, 2]*=weight1
            
        if blend_left[i]!=0:
            weight2 = np.linspace(0, 1, blend_left[i])
            weight2 = np.tile(weight2, (h1, 1))
            tmp[:, :blend_left[i], 0]*=weight2
            tmp[:, :blend_left[i], 1]*=weight2
            tmp[:, :blend_left[i], 2]*=weight2
        
        canvas[yoffsets[i]:yoffsets[i]+h1, xoffsets[i]:xoffsets[i]+w1]+=tmp
    canvas2 = canvas[max(yoffsets):min(yheights), :]
    cv2.imwrite(output, canvas2)
