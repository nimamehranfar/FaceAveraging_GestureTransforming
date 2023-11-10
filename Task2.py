import numpy as np
import cv2 as cv
import dlib
import glob
from matplotlib import pyplot as plt
from face_data import Face1, edges


def affine_transform(mat , base_mat, affine_similarity):

    a , b , c , d = 0 ,0 ,0 ,0
    transform_mat = np.array([ [a, b] , [c ,d] ])
    if affine_similarity == True :
        transform_mat = np.array([ [a, b] , [c ,d] ])
    else:
        transform_mat = np.array([ [a, b] , [ -b ,a] ])

    # F =  ||  mat* transform_mat - base_mat ||
    #  after caoputation derivate
    mat_x1 = mat [ : , 0]
    mat_x2= mat [ : , 1]

    base_mat_y1 = base_mat [ : , 0]
    base_mat_y2 = base_mat [ : , 1]

    A1 = mat_x1.T @ mat_x1
    A2 = mat_x1.T @ mat_x2
    A3 = mat_x1.T @ mat_x2
    A4 = mat_x2.T @ mat_x2
    A = np.array([ [A1,A2] , [A3 ,A4] ])

    W1 = transform_mat[ 0 , : ]
    W2 = transform_mat[ 1 , : ]
    B1_1 = mat_x1.T @ base_mat_y1
    B1_2 = mat_x2.T @ base_mat_y1
    B2_1 = mat_x1.T @ base_mat_y2
    B2_2 = mat_x2.T @ base_mat_y2
    B1 = np.array( [B1_1,B1_2] )
    B2 = np.array( [B2_1,B2_2] )

    a , b = np.linalg.lstsq(A, B1, rcond=None)[0]
    type = "affine , "
    if affine_similarity == False :
        type  = "similarity , "
        transform_mat = np.array([ [a, b] , [-b ,a] ])
    else:
        c , d = np.linalg.lstsq(A, B2, rcond=None)[0]
        transform_mat = np.array([ [a, b] , [ c ,d] ])

    # print ("transform_mat ", type , transform_mat , transform_mat.shape)

    return transform_mat



def plot_face(plt, X, edges, color='b'):
    "plots a face"
    plt.plot(X[:, 0], X[:, 1], 'o', color=color)

    for i, j in edges:
        xi, yi = X[i]
        xj, yj = X[j]

        plt.plot((xi, xj), (yi, yj), '-', color=color)

    plt.axis('square')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


datFile = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datFile)

target_img = cv.imread('Camera Roll/4.jpg', 0)

dets1 = detector(target_img, 1)
Y = []
for k, d in enumerate(dets1):

    shape = predictor(target_img, d)

    Y = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
    for y in Y:
        cv.circle(target_img, (y[0], y[1]), 2, (0, 0, 255))


fnames = glob.glob('Camera Roll/*.jpg')
fnames.sort()


X_Stack = []


count = 1

for fname in fnames:
    count += 1
    print(fname)
    # Open images...
    input_img = cv.imread(fname, 0)
    dets2 = detector(input_img, 1)
    X = []

    for k, d in enumerate(dets2):

        shape = predictor(input_img, d)

        X = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
        for x in X:
            cv.circle(input_img, (x[0], x[1]), 2, (0, 0, 255))
    # cv.imshow('src', input_img)
    affine_similarity = False
    X_transformed =  X @ affine_transform( X ,Y , affine_similarity)


    X_Stack.append( X_transformed )
    # sum += Z

# mean = sum / count

X_Stack = np.stack(X_Stack)
mean1 = X_Stack.mean(axis=0)
# print(np.stack(X_Stack), np.stack(X_Stack).shape)
# print(mean1, mean1.shape)
count, x68, x2 = X_Stack.shape
Z_Stack = []

for i in range(count):
    Z_Stack.append((X_Stack[i] - mean1).ravel())

Z_Stack = np.stack(Z_Stack)
Z_Stack = Z_Stack.T
# print(Z_Stack, Z_Stack.shape)

U, Sig, vh = np.linalg.svd(Z_Stack, full_matrices=True)

# print(U, U.shape)
# print(Sig, Sig.shape)

final = mean1
k_param = 16
for value in range( k_param ):
    rng = np.linspace(-Sig[value], Sig[value], 20 )
    for i in rng:
        plt.cla()
        final = final +  i * (U[:, value].reshape((68, 2)))
        plot_face(plt, final, edges, color='r')
        plt.draw()
        plt.pause(.001)

# plot_face(plt, Y, edges, color='b')
# plt.show()
#
# plot_face(plt, mean1, edges, color='b')
# plt.show()

# cv.imshow('Source image', input_img)
# cv.imshow('Target image', target_img)
# cv.imshow('Warp', affine_dst)
# cv.imshow('Warp2', similar_dst)
# cv.imshow('Warp2', mean)
plt.show()

cv.waitKey()
