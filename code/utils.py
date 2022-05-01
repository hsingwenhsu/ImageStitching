import cv2
import numpy as np
from math import*
import random
from statistics import stdev, mean

def show_img(img):
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 500, 1000)
    cv2.imshow("frame", img)
    cv2.waitKey(0)

# read data 
def read_data(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    path = data[0]
    focal = float(data[1])
    namelist = data[2:]
    return path, focal, namelist

# Image warping
def get_warped(img, f):
    h, w = img.shape[:2]
    warped = np.zeros((h, w, 3), dtype='float32')
    warped[:, :, 1] = 255
    xc = int(w/2)
    yc = int(h/2)
    c1x, c1y = 0, 0
    c2x, c2y = 0, 0
    c3x, c3y = 0, 0
    c4x, c4y = 0, 0
    for i in range(h):
        for j in range(w):
            jlen = j - xc
            ilen = i - yc
            xlen = f*tan(jlen/f)
            ylen = ilen*(xlen**2+f**2)**0.5/f
            x = int(xlen+xc)
            y = int(ylen+yc)
            if ((x>=0 and x<w) and (y>=0 and y<h)):
                warped[i, j] = img[y, x]
            if x==0 and y==0:
                c1x, c1y = j, i
            elif x==w-1 and y==0:
                c2x, c2y = j, i
            elif x==0 and y==h-1:
                c3x, c3y = j, i
            elif x==w-1 and y==h-1:
                c4x, c4y = j, i

    # rhs's ys
    warped2 = warped[max(c1y, c2y):min(c3y, c4y), max(c1x, c3x):min(c2x, c4x)]
    # crop the image

    return warped2

# Harris corner detection
def get_kernel(sigma, size):
    kernel = np.zeros((size, size), dtype='float32')
    if size%2==1:
        center = int((size - 1)/2)
        for i in range(size):
            for j in range(size):
                dist2 = (i-center)**2+(j-center)**2
                exponent = -dist2/2/sigma**2
                val = 1/(sqrt(2*pi)*sigma)*(e**exponent)
                kernel[i, j] = val
    else:
        center1 = (int(size/2)-1, int(size/2)-1) #(x, y)
        center2 = (int(size/2), int(size/2)-1)
        center3 = (int(size/2)-1, int(size/2))
        center4 = (int(size/2), int(size/2))
        
        for i in range(size): 
            for j in range(size):
                # decide which center to use
                if i<=int(size/2)-1 and j<=int(size/2)-1: #center1
                    diff_i = i - center1[1]
                    diff_j = j - center1[0]
                elif i<=int(size/2)-1 and j>=int(size/2): #center2
                    diff_i = i - center2[1]
                    diff_j = j - center2[0]
                elif i>=int(size/2) and j<=int(size/2)-1: #center3
                    diff_i = i - center3[1]
                    diff_j = j - center3[0]
                else: # center 4
                    diff_i = i - center4[1]
                    diff_j = j - center4[0]
                dist2 = diff_i**2+diff_j**2
                exponent = -dist2/2/sigma**2
                val = 1/(sqrt(2*pi)*sigma)*(e**exponent)
                kernel[i, j] = val

    kernel = kernel/np.sum(kernel)
    return kernel

def get_feature_pts(img, k = 0.04):
    # window size = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gsize = 5
    ghalf = int((gsize-1)/2)
    # get the list of coordinates of the feature points
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = gsize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = gsize)
    A = Ix*Ix
    B = Iy*Iy
    C = Ix*Iy
    kernel = get_kernel(1.5, gsize)
    h, w = A.shape[:2]
    R = np.zeros((h, w))

    for i in range(ghalf, h-ghalf):
        for j in range(ghalf, w-ghalf):
            A_tmp = A[i-ghalf:i+ghalf+1, j-ghalf:j+ghalf+1]
            B_tmp = B[i-ghalf:i+ghalf+1, j-ghalf:j+ghalf+1]
            C_tmp = C[i-ghalf:i+ghalf+1, j-ghalf:j+ghalf+1]
            m11 = np.sum(A_tmp*kernel)
            m12 = np.sum(C_tmp*kernel)
            m21 = m12
            m22 = np.sum(B_tmp*kernel)
            M = np.array([[m11, m12], [m21, m22]])
            R[i, j] = np.linalg.det(M)-k*np.square(np.trace(M))
    #we want the top 1000 feature points
    R_rank = R.copy()
    R_rank = R_rank.flatten()
    R_rank.sort()

    feature_pts = [] #(x, y, R value)
    thresh = R_rank[-2000]
    for i in range(h):
        for j in range(w):
            if R[i, j]<thresh:
                R[i, j] = 0
            else:
                feature_pts.append((j, i, R[i, j]))
    # should remove feature points that are too close to each other
    return feature_pts

# descriptors
def get_bin_idx8(rad):
    step = 2*pi/8
    idx = floor(rad/step)
    return idx

def get_descriptors(img, feature_pts):
    descriptors = []
    h, w = img.shape[:2]
    blur_kernel = cv2.getGaussianKernel(17, 5, cv2.CV_64F)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L = cv2.sepFilter2D(gray, -1, blur_kernel, blur_kernel, cv2.BORDER_REFLECT)
    # compute the magnitude
    mag = np.zeros((h, w))
    grad = np.zeros((h, w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            dx = L[i, j+1]-L[i, j-1]
            dy = L[i+1, j]-L[i-1, j]

            mag[i, j] = sqrt(dx**2+dy**2) 
            grad[i, j] = atan(dy/(dx+e-8))

    w_kernel = get_kernel(1.5, 16)
    des_vecs = []
    des_vecs_left = []
    des_vecs_right = []
    for pts in feature_pts:
        x, y = pts[:2]
        # get a patch

        if x-8>=0 and x+7<w and y-8>=0 and y+7<h:
            mpatch = mag[y-8:y+8, x-8:x+8] # a 16*16 patch around a feature point
            gpatch = grad[y-8:y+8, x-8:x+8] # a 16*16 patch around a feature point
            vote = w_kernel*mpatch # the voting weight
            
            ori_hists = [[0]*8, [0]*8, [0]*8, [0]*8,]
            for i in range(8): 
                for j in range(8):
                    rad = gpatch[i, j] # top left
                    bin_idx = get_bin_idx8(rad)
                    ori_hists[0][bin_idx]+=vote[i, j]

                    rad = gpatch[i, j+8] # top right
                    bin_idx = get_bin_idx8(rad)
                    ori_hists[1][bin_idx]+=vote[i, j+8]

                    rad = gpatch[i+8, j] # bottom left
                    bin_idx = get_bin_idx8(rad)
                    ori_hists[2][bin_idx]+=vote[i+8, j]

                    rad = gpatch[i+8, j+8] # bottom right
                    bin_idx = get_bin_idx8(rad)
                    ori_hists[3][bin_idx]+=vote[i+8, j+8]

            ori_hists = sum(ori_hists, [])
            ori_sum = sum(ori_hists)
            des_vec = [item/ori_sum for item in ori_hists]
            # clip the values larger than 0.2
            for i in range(len(des_vec)):
                if des_vec[i]>0.2:
                    des_vec[i] = 0.2

            des_vec = np.array(des_vec)
            if x<w//2:
                des_vecs_left.append((x, y, des_vec))
            else:
                des_vecs_right.append((x, y, des_vec))

            #des_vecs.append((x, y, des_vec))
        des_vecs.sort(key=lambda x:x[0])
        des_vecs_left.sort(key=lambda x:x[0])
        des_vecs_right.sort(key=lambda x:x[0])
    return des_vecs_left, des_vecs_right

# Feature matching
def get_dist(vec1, vec2):
    #diff = (vec1-vec2)*(vec1-vec2)
    #dist = np.sqrt(np.sum(diff))
    dist = np.linalg.norm(np.abs(vec1 - vec2))
    return dist

def get_matched_pts(des_vecs1, des_vecs2):
    matched1 = {} # matched[(x1, y1)]: (x2, y2, dist)
    matched2 = {} # matched[(x2, y2)]: list of (x1, y1, dist)
    matched = {}
    print('# of descriptors 1: '+str(len(des_vecs1)))
    print('# of descriptors 2: '+str(len(des_vecs2)))

    for des_vec1 in des_vecs1:
        x1, y1, vec1 = des_vec1[:3]
        for des_vec2 in des_vecs2:
            x2, y2, vec2 = des_vec2[:3]
            dist = get_dist(vec1, vec2)
            
            if dist<0.07:
                if (x1, y1) in matched1:
                    if dist<matched1[(x1, y1)][2]:
                        matched1[(x1, y1)] = (x2, y2, dist)
                else:
                    matched1[(x1, y1)] = (x2, y2, dist)
    
    for pts in matched1:
        x1, y1 = pts[:2]
        x2, y2, dist = matched1[(x1, y1)][:3]
        if (x2, y2) in matched2:
            matched2[(x2, y2)].append((x1, y1, dist))
        else:
            matched2[(x2, y2)] = [(x1, y1, dist)]
        
    for pts in matched2:
        x2, y2 = pts[:2]
        matched2[pts].sort(key=lambda x:x[2]) #sort the list
        #print('matched 2: ', matched2[pts])
        x1, y1, dist = matched2[pts][0][:3]
        matched[(x1, y1)] = (x2, y2, dist)

    ranked = sorted(matched.items(), key=lambda k: k[1][2])

    top = len(ranked)
    matched_pts = []
    #if len(ranked)<top:
    #    top = len(ranked)

    for i in range(top):
        x1, y1 = ranked[i][0][:2]
        x2, y2 = ranked[i][1][:2]
        matched_pts.append((x1, y1, x2, y2))
    
    return matched_pts
    
def ransac(matched):
    print('Number of matched points', len(matched))
    P = 0.99999
    n = 4
    p = 0.5
    thresh = 10 # tbd
    k = int(log(1-P)/log(1-p**n)) # number of iterations
    k = k*2
    pt_num = len(matched)
   
    match_cnt = {} # match_cnt[(m1, m2)] = the number of inliers
    inlier_dict = {}
    for i in range(k):
        # sample n points
        samples = random.sample(matched, n)
        notsamples = matched.copy()
        for j in range(n):
            notsamples.remove(samples[j])
        
        m1 = 0
        m2 = 0
        
        # compute m1 and m2
        for j in range(n):
            m1+=(samples[j][0]-samples[j][2])
            m2+=(samples[j][1]-samples[j][3])
        
        m1 = m1/n
        m2 = m2/n
        
        if (m1, m2) not in match_cnt:
            for j in range(n):
                inlier_dict[(m1, m2)] = [(samples[j][0], samples[j][1], samples[j][2], samples[j][3], 0)]
            match_cnt[(m1, m2)] = n

        for j in range(pt_num-n):
            x1, y1, x2, y2 = notsamples[j][:4]
            dist_x = x1-x2-m1
            dist_y = y1-y2-m2
            dist = sqrt(dist_x**2+dist_y**2)

            if dist < thresh:
                match_cnt[(m1, m2)]+=1
                inlier_dict[(m1, m2)].append((x1, y1, x2, y2, dist))
               
                    
        # find out the (m1, m2)
    ranked = sorted(match_cnt.items(), key=lambda x:x[1], reverse=True)
    
    max_m1, max_m2 = ranked[0][0][:2]

    final_matched = inlier_dict[(max_m1, max_m2)] # a list of matched inliers   
    final_matched.sort(key=lambda x:x[4])
    
    return final_matched


# for image stitching
def get_stitch_info(final_matched, xoffset_pre, yoffset_pre):
    # xoffset_pre is the offset of the left image
    # compute the offset for the two images
    #best_x1, best_x2, best_y1, best_y2 = final_matched[0][:4]
    #matched_left = []
    #matched_right = []
    xleft = 0 # the left most point in the left image
    xright = 0 # the right most point in the right image
    """
    for pts in final_matched:
        x1, y1, x2, y2 = pts[:4]
        matched_left.append((x1, y1))
        matched_right.append((x2, y2))
    """
    #matched_left.append((best_x1, best_y1))
    #matched_right.append((best_x2, best_y2))
    #matched_left.sort(key=lambda x:x[1], reverse=True)
    #matched_right.sort(key=lambda x:x[1])
    #xleft = matched_left[0][1]
    #xright = matched_right[0][1]

    x1, y1, x2, y2 = final_matched[0][:4]
    # x offset of the right image
    xoffset = xoffset_pre+x1-x2
    yoffset = yoffset_pre+y1-y2
    
    return xoffset, yoffset
    


