import json

def square_root(x):
    if x > 0:
        return x ** (1 / 2)
    else:
        return -1 * (-x) ** (1 / 2)

def power(x, level):
    return x ** level

def euclidean_distance(A, B):
    if len(A) != len(B):
        return -1
    else:
        length = len(A)
        total = 0
        for i in range(length):
            total += power(B[i] - A[i], 2)
        distance = square_root(total)
        return distance

def dot_product(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1

    total = 0
    for i in range(len(vector1)):
        total += vector1[i][1] * vector2[i][1]
    return total

def vector_magnitude(vector):
    total = 0
    for i in range(len(vector)):
        total += vector[i][1] ** 2
    return total ** (1 / 2)

def cosine_similarity(vector1, vector2):
    return dot_product(vector1, vector2) / (vector_magnitude(vector1) * vector_magnitude(vector2))

def calculate_vector(A, B):
    element = [abs(A[0] - B[0]), abs(A[1] - B[1])]
    return element

def load_face_vector(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def save_face_vector(face_points, path):
    with open(path, 'w') as outfile:
        json.dump(face_points, outfile)
