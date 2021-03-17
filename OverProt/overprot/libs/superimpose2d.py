import numpy  # type: ignore
from typing import Any, List, Dict, Tuple


INT = numpy.int_  # type: ignore


def optimal_rotation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False) -> numpy.ndarray:
    ''' Find the optimal rotation matrix for 2D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    '''
    # norm_coef = numpy.sqrt(2 / (numpy.mean(A**2) + numpy.mean(B**2)))
    # A = A * norm_coef
    # B = B * norm_coef
    C = numpy.matmul(A, B.transpose())
    if allow_mirror and numpy.linalg.det(C) < 0:  # type: ignore
        # print('mirror')
        dmir1 = C[1,0] + C[0,1]
        dmir2 = C[0,0] - C[1,1] 
        sumsq_d = dmir1*dmir1 + dmir2*dmir2
        if sumsq_d == 0:
            return numpy.eye(2)
        q_norm = numpy.sqrt(1 / sumsq_d)
        xmir = dmir1 * q_norm
        ymir = dmir2 * q_norm
        return numpy.array([[ymir, xmir], [xmir, -ymir]])
    else:
        d1 = C[0,0] + C[1,1]
        d2 = C[1,0] - C[0,1]
        sumsq_d = d1*d1 + d2*d2
        if sumsq_d == 0:
            return numpy.eye(2)
        q_norm = numpy.sqrt(1 / sumsq_d)
        x = d1 * q_norm
        y = d2 * q_norm
        return numpy.array([[x, y], [-y, x]])

def optimal_rotation_translation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False) -> Tuple[numpy.ndarray, numpy.ndarray]:
    ''' Find the optimal rotation matrix R and translation vector t for 2D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    A_superimposed = R * A + t
    '''
    cA = numpy.mean(A, axis=1, keepdims=True)
    cB = numpy.mean(B, axis=1, keepdims=True)
    R = optimal_rotation(A - cA, B - cB, allow_mirror=allow_mirror)
    t = numpy.matmul(R, -cA) + cB
    return R, t

def optimal_rotation_translation_rmsd(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    ''' Work as optimal_rotation_translation() but return a triple (rotation, translation, RMSD).
    '''
    R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
    A = rotate_and_translate(A, R, t)
    RMSD = rmsd(A, B, superimpose=False)
    return R, t, RMSD

def rotate_and_translate(A: numpy.ndarray, rotation: numpy.ndarray, translation: numpy.ndarray) -> numpy.ndarray:
    ''' Applie rotation (or improper rotation) matrix 'rotation' and translation vector 'translation'
    to a 2D object A,
    where columns of A are coordinates of individual points.
    A_result = R * A + t
    '''
    if isinstance(A, dict):
        # on dictionaries:
        matrix = numpy.array(list(A.values())).transpose()
        matrix = rotate_and_translate(matrix, rotation, translation)
        return { name: list(matrix[:,i]) for i, name in enumerate(A.keys()) }
    else:
        # on matrices:
        return numpy.matmul(rotation, A) + translation

def rmsd(A: numpy.ndarray, B: numpy.ndarray, superimpose: bool=True, allow_mirror: bool=False) -> float:
    ''' Calculate RMSD between 2D objects A and B.
    If superimpose == False, skip superimposition and calculates RMSD as is.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    '''
    if superimpose:
        R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
        A = rotate_and_translate(A, R, t)
    diff = A - B
    n = A.shape[1]
    sumsq = numpy.sum(diff**2)
    return numpy.sqrt(sumsq / n)

def distinct(lst):
    seen = set()
    for x in lst:
        if x not in seen:
            seen.add(x)
            yield x

def is_clockwise(A):
    R = A[:,1:] + A[:,:-1]
    V = A[:,1:] - A[:,:-1]
    M = R[0,:] * V[1,:] - R[1,:] * V[0,:]
    return sum(M) <= 0

def laying_rotation(A, force_left_to_right=False, force_clockwise=False):
    ''' Return such rotation matrix R that the first PCA component of R*A is parallel with x-axis.
    Require that A be centered (centroid == [0, 0])!
    If force_left_to_right == True, provide such R that first point of R*A is more left than the last point, otherwise provide rotation with minimum rotation angle.
    If force_clockwise == True, provide such (possibly improper) rotation that direction of the points is roughly clockwise, otherwise provide always proper rotation.
    '''
    C = numpy.matmul(A, A.transpose())
    c11 = C[0,0]
    c12 = C[0,1]
    c22 = C[1,1]
    D = (c11 - c22)**2 + 4 * c12**2
    if D < 0:
        print('Warning: negative determinant, using zero instead')
        D = 0
    if D == 0:
        return numpy.eye(2)
    else:
        lambda1 = 0.5 * (c11 + c22 + numpy.sqrt(D))
        if abs(c22-lambda1) >= abs(c11-lambda1):
            (a, b) = (lambda1 - c22, c12) 
        else:
            (a, b) = (c12, lambda1 - c11) 
        q = (a**2 + b**2)**(-0.5)
        (a, b) = (q * a, q * b)
        if a < 0:
            (a, b) = (-a, -b)
        R = numpy.array([[a, b], [-b, a]])

        # print(force_left_to_right,  force_clockwise)
        # print(numpy.matmul(R, A[:,0])[0], numpy.matmul(R, A[:,-1])[0], numpy.matmul(R, A[:,0])[0] > numpy.matmul(R, A[:,-1])[0])
        # print(not is_clockwise(A))
        if force_left_to_right and numpy.matmul(R, A[:,0])[0] > numpy.matmul(R, A[:,-1])[0]:
            R = -R
        if force_clockwise and not is_clockwise(A):
            R[1,:] = -R[1,:]
        return R

def multi_superimpose(objects: List[Dict[Any, List[float]]], allow_mirror=False, rmsd_epsilon=0.001, max_iter=100, plot=False, verbose=False) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    ''' Find the optimal rotation matrix R and translation vector t for each of n objects.
    Each object is defined as a dictionary where keys are names of points and values are their 2D coordinates.
    Matching of the points from different objects is based on their names.
    e.g.:
        multi_superimpose([{'A':[0,1], 'B':[1,1], 'C':[1,0]}, {'A':[2,2], 'C':[1,1.1], 'D':[1.5,2]}], allow_mirror=True, plot=True)
    '''
    # all_names = list({ key for obj in objects for key in obj.keys() })
    all_names = list(distinct( key for obj in objects for key in obj.keys() ))
    name2global_index = { name: j for j, name in enumerate(all_names) }
    local2globals = []
    global2locals = []
    n_objects = len(objects)
    n_points = len(all_names)
    matrices = [] 
    for i, obj in enumerate(objects):
        matrix = numpy.array([ xy for xy in obj.values() ]).transpose()
        matrices.append(matrix)
        loc2glob = [ name2global_index[name] for name in obj.keys() ]
        glob2loc = [-1] * n_points
        for loc, glob in enumerate(loc2glob):
            glob2loc[glob] = loc
        local2globals.append(loc2glob)
        global2locals.append(glob2loc)
    i_largest = max(range(n_objects), key=lambda i: len(objects[i]))
    i_consensus = i_largest
    consensus_matrix = matrices[i_consensus]
    consensus_loc2glob = local2globals[i_consensus]
    consensus_glob2loc = global2locals[i_consensus]

    if plot:
        from matplotlib import pyplot as plt
        for i in range(n_objects):
            plt.plot(matrices[i][0,:], matrices[i][1,:], 'o')
        plt.show()

    rotations_translations = [ (numpy.eye(2), numpy.zeros((2,1)) ) for i in range(n_objects) ]
    last_total_rmsd = numpy.inf
    last_used_points = 0
    iteration = 0   
    while True:
        center = numpy.mean(consensus_matrix, axis=1, keepdims=True)
        consensus_matrix = consensus_matrix - center
        Lay = laying_rotation(consensus_matrix, force_left_to_right=True, force_clockwise=allow_mirror)
        consensus_matrix = numpy.matmul(Lay, consensus_matrix)
        total_RMSD = 0.0
        sum_matrix = numpy.zeros((2, n_points))
        point_usages = numpy.zeros((n_points,), dtype=INT)
        n_aligned_objects = 0
        for i in range(n_objects):
            common_points = [ g for g in range(n_points) if consensus_glob2loc[g]>=0 and global2locals[i][g]>=0]
            if len(common_points) < 2:
                # print(f'Warning: object {i} has less than 2 common points with consensus')
                continue
            A_locals = [ global2locals[i][g] for g in common_points ]
            B_locals = [ consensus_glob2loc[g] for g in common_points ]
            A = matrices[i][:, A_locals]  # type: ignore  
            B = consensus_matrix[:, B_locals]  # type: ignore  
            R, t, RMSD = optimal_rotation_translation_rmsd(A, B, allow_mirror=allow_mirror)  # type: ignore
            total_RMSD += RMSD
            point_usages[local2globals[i]] += 1  
            sum_matrix[:, local2globals[i]] += rotate_and_translate(matrices[i], R, t)  # type: ignore
            rotations_translations[i] = (R, t)
            n_aligned_objects += 1
            A_full = rotate_and_translate(matrices[i], R, t)
            if plot: plt.plot(A_full[0,:], A_full[1,:], 'o')
        used_points = [ g for g, usage in enumerate(point_usages) if usage > 0 ]
        consensus_matrix = sum_matrix[:, used_points] / point_usages[used_points]  # type: ignore
        consensus_loc2glob = used_points
        consensus_glob2loc = [-1] * n_points
        for loc, glob in enumerate(consensus_loc2glob):
            consensus_glob2loc[glob] = loc
        # rel_RMSD = total_RMSD / len(used_points)
        if verbose: print(f'Iteration {iteration}: {n_aligned_objects} aligned objects, {len(used_points)} used points, RMSD {total_RMSD}')
        if plot: 
            plt.plot(consensus_matrix[0,:], consensus_matrix[1,:], 'x-k')
            plt.plot(consensus_matrix[0,0], consensus_matrix[1,0], 'ok')
            plt.title(f'Iteration {iteration}: {n_aligned_objects} aligned objects, {len(used_points)} used points, RMSD {total_RMSD}')
            plt.axis('equal')  # type: ignore
            plt.show()
        if iteration >= max_iter or (last_used_points == len(used_points) and 0 <= last_total_rmsd - total_RMSD <= rmsd_epsilon):
            break
        else:
            iteration += 1
            last_total_rmsd = total_RMSD
            last_used_points = len(used_points)
    return rotations_translations
  

# Testing:

def random_transform_plus_noise(A, noise=0.01):
    A = A + numpy.random.normal(scale=noise, size=A.shape)
    if numpy.random.random() >= 0.5:
        A[0,:] = -A[0,:]
    theta = 2 * numpy.pi * numpy.random.random()
    R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
    t0 = 10 * (numpy.random.random([2,1]) - 0.5)
    t0 = numpy.array([[4], [2]])
    A = numpy.matmul(R0, A) + t0
    return A

def test1(theta=None):
    from matplotlib import pyplot as plt
    B = 3 * numpy.random.random(size=(2,6))
    A = B + numpy.random.normal(scale=0.1, size=B.shape)
    if numpy.random.random() >= 0.5:
        A[0,:] = -A[0,:]
    if theta is None:
        theta = 2 * numpy.pi * numpy.random.random()
    R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
    t0 = 10 * (numpy.random.random([2,1]) - 0.5)
    t0 = numpy.array([[4], [2]])
    A = numpy.matmul(R0, A) + t0
    R, t = optimal_rotation_translation(A, B, allow_mirror=True)
    At = numpy.matmul(R, A) + t
    # print(theta*180/numpy.pi)
    print(rmsd(A, B, superimpose=False), rmsd(A, B, superimpose=True, allow_mirror=True))
    plt.plot(B[0,:], B[1,:], 'o-')
    plt.plot(A[0,:], A[1,:], 'o-')
    plt.plot(At[0,:], At[1,:], 'o-')
    plt.axis('equal')
    plt.show()

def test2():
    n = 100
    k = 20
    k_ = 10
    X = numpy.random.random(size=(2,k))
    X[1,:] = 2 * X[1,:]
    Xs = [ random_transform_plus_noise(X, noise=0.02) for i in range(n) ]
    objects = [ { j: Xs[i][:,j] for j in numpy.random.choice(range(k), size=k_, replace=False) } for i in range(n) ]
    multi_superimpose(objects, allow_mirror=True, verbose=True, plot=True)

def test3():
    from matplotlib import pyplot as plt
    B = 2 * numpy.random.random(size=(2,5))
    B = B - B.mean(axis=1, keepdims=True)
    B[0,:] = 3 * B[0,:]
    theta = 2 * numpy.pi * numpy.random.random()
    R0 = numpy.array([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
    A = numpy.matmul(R0, B)

    R = laying_rotation(A, force_left_to_right=True, force_clockwise=True)
    print(R)
    At = numpy.matmul(R, A)
    plt.plot(A[0,:], A[1,:], 'o-')
    plt.plot(At[0,:], At[1,:], 'o-')
    plt.plot(A[0,0], A[1,0], 'xk')
    plt.plot(At[0,0], At[1,0], 'xk')
    plt.axis('equal')
    plt.show()

def demo():
    from matplotlib import pyplot as plt
    # Zadefinujeme objekty:
    # Mame 2 objekty, prvy obsahuje helixy H1, H2, H3, H4, druhy len H1, H2, H3. 
    # Pre jednoduchost vsetky helixy maju vahu 1 (1 atom).
    object_A = {'H1_0': [0,0], 'H2_0': [0,2], 'H3_0': [1,2], 'H4_0': [1,2.5]}
    object_B = {'H1_0': [1,0], 'H2_0': [3.2,0.2], 'H3_0': [3,1]}

    # Spustime zarovnanie (ziskame zarovnavacie rotacie a translacie):
    rotations_translations = multi_superimpose([object_A, object_B], allow_mirror=True)

    # Ziskame z objektov stlpcove matice:
    A = numpy.array(list(object_A.values())).transpose()
    B = numpy.array(list(object_B.values())).transpose()
    print('A =', A, 'B =', B, sep='\n')

    # Aplikujeme zarovnavacie rotacie a translacie:
    R, t = rotations_translations[0]
    A_aligned = rotate_and_translate(A, R, t)
    object_A_aligned = rotate_and_translate(object_A, R, t)
    A_aligned = numpy.array(list( list(c) for c in object_A_aligned.values() )).transpose()
    print(object_A)
    print(object_A_aligned)
    R, t = rotations_translations[1]
    B_aligned = rotate_and_translate(B, R, t)

    # Vykreslime:
    plt.subplot(1, 2, 1)
    plt.title('Pred zarovnanim')
    plt.plot(A[0,:], A[1,:], 'o-')
    plt.plot(B[0,:], B[1,:], 'o-')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title('Po zarovnani')
    plt.plot(A_aligned[0,:], A_aligned[1,:], 'o-')
    plt.plot(B_aligned[0,:], B_aligned[1,:], 'o-')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    demo()
    pass