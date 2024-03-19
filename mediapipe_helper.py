import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def draw_results(results, image):
    """
    Draws hand landmarks on the given image.

    Args:
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The hand landmarks detected by Mediapipe.
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with hand landmarks drawn.
    """
    annotated_image = image.copy()

    if results is None or results.multi_hand_landmarks is None:
        return annotated_image

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        return annotated_image


def visualize_keypoints(results):
    """
    Visualizes the hand keypoints in 3D space.

    Args:
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The hand landmarks detected by Mediapipe.
    """
    for hand_world_landmarks in results.multi_hand_world_landmarks:
        mp_drawing.plot_landmarks(
            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5
        )


def landmark_to_feature(results):
    """
    Converts hand landmarks to a feature vector.

    Args:
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The hand landmarks detected by Mediapipe.

    Returns:
        list: The feature vector representing the hand landmarks.
    """
    feature_vector = []

    if not is_hand_present(results):
        return feature_vector

    for hand_landmarks in results.multi_hand_landmarks:
        for keypoint in mp_hands.HandLandmark:
            feature_vector.append(hand_landmarks.landmark[keypoint].x)
            feature_vector.append(hand_landmarks.landmark[keypoint].y)
            feature_vector.append(hand_landmarks.landmark[keypoint].z)

    return feature_vector


def is_hand_present(results):
    """
    Checks if a hand is present in the results.

    Args:
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The hand landmarks detected by Mediapipe.

    Returns:
        bool: True if a hand is present, False otherwise.
    """
    return (
        results is not None
        and results.multi_hand_landmarks is not None
        and len(results.multi_handedness) > 0
    )
