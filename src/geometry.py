import numpy as np

def calculate_distance(point_a, point_b):
    x_difference = point_a.x - point_b.x
    y_difference = point_a.y - point_b.y
    z_difference = point_a.z - point_b.z

    distance = np.sqrt(
        (x_difference ** 2) +
        (y_difference ** 2) +
        (z_difference ** 2)
    )

    return distance


def get_palm_size(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_finger_base = hand_landmarks.landmark[9]

    palm_size = calculate_distance(wrist, middle_finger_base)

    return palm_size


def get_normalized_distance(point_a, point_b, hand_landmarks):
    raw_distance = calculate_distance(point_a, point_b)
    palm_size = get_palm_size(hand_landmarks)
    normalized_distance = raw_distance / palm_size

    return normalized_distance


def calculate_angle(point_a, point_b, point_c):
    a = np.array([point_a.x, point_a.y, point_a.z])
    b = np.array([point_b.x, point_b.y, point_b.z])
    c = np.array([point_c.x, point_c.y, point_c.z])

    vector_1 = a - b
    vector_2 = c - b

    dot_product = np.dot(vector_1, vector_2)

    length_1 = np.linalg.norm(vector_1)
    length_2 = np.linalg.norm(vector_2)

    cosine_angle = dot_product / (length_1 * length_2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def get_finger_flexion(hand_landmarks, finger_name):
    finger_landmark_map = {
        "index":  [5, 6, 8],
        "middle": [9, 10, 12],
        "ring":   [13, 14, 16],
        "pinky":  [17, 18, 20],
    }

    landmark_indices = finger_landmark_map[finger_name]

    base_knuckle = hand_landmarks.landmark[landmark_indices[0]]
    middle_knuckle = hand_landmarks.landmark[landmark_indices[1]]
    fingertip = hand_landmarks.landmark[landmark_indices[2]]

    flexion_angle = calculate_angle(base_knuckle, middle_knuckle, fingertip)

    return flexion_angle