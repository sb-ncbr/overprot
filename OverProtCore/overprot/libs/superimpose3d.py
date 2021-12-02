import numpy
from numpy import linalg
from typing import Any, List, Dict, Tuple

def optimal_rotation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False, weights: numpy.ndarray=None) -> numpy.ndarray:
    ''' A, B - matrices 3*n, weights - vector n, result - matrix 3*3
    Find the optimal rotation matrix for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    '''
    if weights is not None:
        A = A * weights.reshape((1, -1))
    H = A @ B.transpose()
    U, S, Vh = linalg.svd(H)
    R = (U @ Vh).transpose()
    if not allow_mirror and numpy.linalg.det(R) < 0:  # type: ignore  # mypy doesn't know .det
        Vh[-1,:] = -Vh[-1,:]
        R = (U @ Vh).transpose()
    return R

def optimal_rotation_translation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False, weights: numpy.ndarray=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''  A, B - matrices 3*n, weights - vector n, result - (matrix 3*3, matrix 3*1)
    Find the optimal rotation matrix R and translation vector t for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    A_superimposed = R * A + t
    '''
    if weights is not None:
        sumW = weights.sum()
        cA = (A * weights).sum(axis=1, keepdims=True) / sumW
        cB = (B * weights).sum(axis=1, keepdims=True) / sumW
    else:
        cA = numpy.mean(A, axis=1, keepdims=True)
        cB = numpy.mean(B, axis=1, keepdims=True)
    R = optimal_rotation(A - cA, B - cB, allow_mirror=allow_mirror, weights=weights)
    t = numpy.matmul(R, -cA) + cB
    return R, t

def rotate_and_translate(A: numpy.ndarray, rotation: numpy.ndarray = numpy.eye(3), translation: numpy.ndarray = numpy.zeros((3,1))) -> numpy.ndarray:
    ''' Applies rotation (or improper rotation) matrix 'rotation' and translation vector 'translation'
    to a 2D object A,
    where columns of A are coordinates of individual points.
    A_result = R * A + t
    '''
    return rotation @ A + translation

def rmsd(A: numpy.ndarray, B: numpy.ndarray, superimpose: bool=True, allow_mirror: bool=False) -> float:
    ''' Calculates RMSD between 2D objects A and B.
    If superimpose == False, skips superimposition and calculates RMSD as is.
    If allow_mirror == True, allows also improper rotation (i.e. mirroring + rotation).
    '''
    if superimpose:
        R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
        A = rotate_and_translate(A, R, t)
    diff = A - B
    n = A.shape[1]
    sumsq = numpy.sum(diff**2)
    return numpy.sqrt(sumsq / n)

def optimal_rotation_translation_rmsd(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    ''' Works as optimal_rotation_translation() but returns a triple (rotation, translation, RMSD).
    '''
    R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
    A = rotate_and_translate(A, R, t)
    RMSD = rmsd(A, B, superimpose=False)
    return R, t, RMSD

def laying_rotation_translation(A: numpy.ndarray, return_laid_coords: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''Return rotation and translation matrices which center A and align the PCA1, 2, 3 with axes x, y, z.
    If return_laid_coords==True, also return the transformed matrix A.
    One of 4 possible results is selected so that: 
    1) starting coordinates tend to be more left-top (x < y), ending more right-bottom (x > y), 
    2) starting and ending coordinates tend to be more in front (z > 0), middle more behind (z < 0).'''
    cA = numpy.mean(A, axis=1, keepdims=True)
    R = laying_rotation(A - cA)
    t = -R @ cA
    return R, t

def laying_rotation(A: numpy.ndarray) -> numpy.ndarray:
    '''Return rotation matrix which aligns the PCA1, 2, 3 with axes x, y, z.
    Centered input matrix is expected.
    One of 4 possible results is selected so that: 
    1) starting coordinates tend to be more left-top (x < y), ending more right-bottom (x > y), 
    2) starting and ending coordinates tend to be more in front (z > 0), middle more behind (z < 0).'''
    assert A.shape[0] == 3
    # assert A.shape[1] >= 3  # ?
    n = A.shape[1]
    U, S, Vh = numpy.linalg.svd(A)
    R = U.T
    if numpy.linalg.det(R) < 0:  # type: ignore  # avoid improper rotation (mirroring)
        R[-1,:] *= -1
    slope = numpy.linspace(-1, 1, n)
    vee_slope = v_slope(n)
    # A_rot = R @ A
    # left_to_right = numpy.dot(A_rot[0, :], slope) > 0
    # front_to_front = numpy.dot(A_rot[2, :], vee_slope) > 0
    # if left_to_right and not front_to_front:
    #     R = numpy.diag([1, -1, -1]) @ R  # rotate around x
    # elif not left_to_right and front_to_front:
    #     R = numpy.diag([-1, -1, 1]) @ R  # rotate around z
    # elif not left_to_right and not front_to_front:
    #     R = numpy.diag([-1, 1, -1]) @ R  # rotate around y
    A_rot = R @ A
    front_to_front = numpy.dot(A_rot[2, :], vee_slope) > 0
    if not front_to_front:  # rotate around x
        R = numpy.diag([1, -1, -1]) @ R  # type: ignore
    A_rot = R @ A
    lefttop_to_rightbottom = numpy.dot(A_rot[0, :] - A_rot[1, :], slope) > 0
    if not lefttop_to_rightbottom:  # rotate around z
        R = numpy.diag([-1, -1, 1]) @ R  # type: ignore
    return R

def v_slope(n: int) -> numpy.ndarray:
    '''Return an array with values in V shape, i.e. starting with +x, linearly decreasing to -y in the middle, and then increasing back to +x.
    The values have mean 0 and stdev 1.'''
    n_half = (n+1)//2
    result = numpy.empty(n)
    result[:n_half] = numpy.linspace(1, -1, n_half)
    result[-n_half:] = numpy.linspace(-1, 1, n_half)
    result -= numpy.mean(result)  # type: ignore
    result /= numpy.std(result)
    return result

def dumb_align(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    '''From matrices A (shape 3*m) and B (shape 3*n) select submatrices A', B' (shape 3*k, k=min(m,n)) minimizing RMSD(A', B').
    Return ... rotation, translation matrix and RMSD. A is the mobile, B is the target matrix.'''
    _, m = A.shape  # mobile
    _, n = B.shape  # target
    k = min(m, n)
    min_shift, max_shift = sorted((0, n - m))
    tranformations = []
    for shift in range(min_shift, max_shift+1):
        if shift >= 0:
            rot, trans, rmsd_ = optimal_rotation_translation_rmsd(A[:, 0:k], B[:, shift:k+shift], allow_mirror=allow_mirror)
        else:
            rot, trans, rmsd_ = optimal_rotation_translation_rmsd(A[:, -shift:k-shift], B[:, 0:k], allow_mirror=allow_mirror)
        tranformations.append((rmsd_, shift, rot, trans))
    rmsd_, shift, rot, trans = min(tranformations)
    return (rot, trans, rmsd_)

#region Testing

# def random_transform_plus_noise(A, noise=0.01):
#     A = A + numpy.random.normal(scale=noise, size=A.shape)
#     if numpy.random.random() >= 0.5:
#         A[0,:] = -A[0,:]
#     theta = 2 * numpy.pi * numpy.random.random()
#     R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
#     t0 = 10 * (numpy.random.random([2,1]) - 0.5)
#     t0 = numpy.array([[4], [2]])
#     A = numpy.matmul(R0, A) + t0
#     return A

# def test1(theta=None):
#     from matplotlib import pyplot as plt
#     B = 3 * numpy.random.random(size=(2,6))
#     A = B + numpy.random.normal(scale=0.1, size=B.shape)
#     if numpy.random.random() >= 0.5:
#         A[0,:] = -A[0,:]
#     if theta is None:
#         theta = 2 * numpy.pi * numpy.random.random()
#     R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
#     t0 = 10 * (numpy.random.random([2,1]) - 0.5)
#     t0 = numpy.array([[4], [2]])
#     A = numpy.matmul(R0, A) + t0
#     R, t = optimal_rotation_translation(A, B, allow_mirror=True)
#     At = numpy.matmul(R, A) + t
#     # print(theta*180/numpy.pi)
#     print(rmsd(A, B, superimpose=False), rmsd(A, B, superimpose=True, allow_mirror=True))
#     plt.plot(B[0,:], B[1,:], 'o-')
#     plt.plot(A[0,:], A[1,:], 'o-')
#     plt.plot(At[0,:], At[1,:], 'o-')
#     plt.axis('equal')
#     plt.show()

# def test2():
#     n = 100
#     k = 20
#     k_ = 10
#     X = numpy.random.random(size=(2,k))
#     X[1,:] = 2 * X[1,:]
#     Xs = [ random_transform_plus_noise(X, noise=0.02) for i in range(n) ]
#     objects = [ { j: Xs[i][:,j] for j in numpy.random.choice(range(k), size=k_, replace=False) } for i in range(n) ]
#     multi_superimpose(objects, allow_mirror=True, verbose=True, plot=True)

# def test3():
#     from matplotlib import pyplot as plt
#     B = 2 * numpy.random.random(size=(2,5))
#     B = B - B.mean(axis=1, keepdims=True)
#     B[0,:] = 3 * B[0,:]
#     theta = 2 * numpy.pi * numpy.random.random()
#     R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
#     A = numpy.matmul(R0, B)

#     R = laying_rotation(A, force_left_to_right=True, force_clockwise=True)
#     print(R)
#     At = numpy.matmul(R, A)
#     plt.plot(A[0,:], A[1,:], 'o-')
#     plt.plot(At[0,:], At[1,:], 'o-')
#     plt.plot(A[0,0], A[1,0], 'xk')
#     plt.plot(At[0,0], At[1,0], 'xk')
#     plt.axis('equal')
#     plt.show()

# def demo():
#     from matplotlib import pyplot as plt
#     # Zadefinujeme objekty:
#     # Mame 2 objekty, prvy obsahuje helixy H1, H2, H3, H4, druhy len H1, H2, H3. 
#     # Pre jednoduchost vsetky helixy maju vahu 1 (1 atom).
#     object_A = {'H1_0': [0,0], 'H2_0': [0,2], 'H3_0': [1,2], 'H4_0': [1,2.5]}
#     object_B = {'H1_0': [1,0], 'H2_0': [3.2,0.2], 'H3_0': [3,1]}

#     # Spustime zarovnanie (ziskame zarovnavacie rotacie a translacie):
#     rotations_translations = multi_superimpose([object_A, object_B], allow_mirror=True)

#     # Ziskame z objektov stlpcove matice:
#     A = numpy.array(list(object_A.values())).transpose()
#     B = numpy.array(list(object_B.values())).transpose()
#     print('A =', A, 'B =', B, sep='\n')

#     # Aplikujeme zarovnavacie rotacie a translacie:
#     R, t = rotations_translations[0]
#     A_aligned = rotate_and_translate(A, R, t)
#     object_A_aligned = rotate_and_translate(object_A, R, t)
#     A_aligned = numpy.array(list( list(c) for c in object_A_aligned.values() )).transpose()
#     print(object_A)
#     print(object_A_aligned)
#     R, t = rotations_translations[1]
#     B_aligned = rotate_and_translate(B, R, t)

#     # Vykreslime:
#     plt.subplot(1, 2, 1)
#     plt.title('Pred zarovnanim')
#     plt.plot(A[0,:], A[1,:], 'o-')
#     plt.plot(B[0,:], B[1,:], 'o-')
#     plt.axis('equal')

#     plt.subplot(1, 2, 2)
#     plt.title('Po zarovnani')
#     plt.plot(A_aligned[0,:], A_aligned[1,:], 'o-')
#     plt.plot(B_aligned[0,:], B_aligned[1,:], 'o-')
#     plt.axis('equal')
#     plt.show()

# if __name__ == "__main__":
#     demo()
#     pass

#endregion