import cv2
import numpy as np
import pywt
from skimage.transform import radon
from deepface import DeepFace
from mtcnn import MTCNN

def case_1(group_image_path, dress_image_path, face_image_path):
    group_image = cv2.imread(group_image_path)
    dress_image = cv2.imread(dress_image_path)

    # Ensure images are loaded correctly
    if group_image is None or dress_image is None:
        print("Error loading images. Check the paths.")
        return

    # 1. Color Detection and Matching
    hist_group = color_histogram(group_image)
    hist_dress = color_histogram(dress_image)
    color_match_score = compare_histograms(hist_group, hist_dress)
    print(f"Color Match Score: {color_match_score}")

    # 2. Pattern Detection using Haar Wavelet Transform
    haar_group = haar_wavelet_transform(cv2.cvtColor(group_image, cv2.COLOR_BGR2GRAY))
    haar_dress = haar_wavelet_transform(cv2.cvtColor(dress_image, cv2.COLOR_BGR2GRAY))

    # 3. Pattern Matching using SIFT
    good_matches, kp1, kp2 = match_sift_features(group_image, dress_image)
    print(f"Number of SIFT Matches: {len(good_matches)}")

    # Draw matched area on the group image
    matched_region = draw_matched_area(group_image, kp1, good_matches)

    # Detect faces in the original image using MTCNN with dynamic adjustment
    detected_faces, faces_coords = detect_faces(group_image, matched_region)

    # Iterate over each detected face and match with the given face image
    for idx, (face, coords) in enumerate(zip(detected_faces, faces_coords)):
        # Save detected face as a temporary file
        face_temp_path = f"detected_face_{idx}.jpg"
        cv2.imwrite(face_temp_path, face)

        # Use DeepFace to verify the face with another face image, and enforce detection
        try:
            result = DeepFace.verify(face_temp_path, face_image_path, enforce_detection=False)
            print(f"Face {idx + 1} - Is verified: {result['verified']}")

            if result["verified"] == True:
                # Draw a rectangle around the verified face
                x, y, w, h = coords
                cv2.rectangle(group_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(group_image, "Verified", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Face {idx + 1} - Error: {str(e)}")

    # Show the result with verified face highlighted
    cv2.imshow('Matched Region with Verified Faces', group_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to calculate the color histogram
def color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to compare histograms (color match)
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to perform Haar wavelet transform for pattern detection
def haar_wavelet_transform(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL

# Function to perform radon transform
def radon_transform(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram

# Function to match SIFT features
def match_sift_features(group_image, dress_image):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(group_image, None)
    kp2, des2 = sift.detectAndCompute(dress_image, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = sorted(matches, key=lambda x: x.distance)[:10]  # Take top 10 matches

    return good_matches, kp1, kp2

# Function to draw matched area
def draw_matched_area(group_image, kp1, good_matches, height_increase=20):
    matched_region = []
    for match in good_matches:
        img1_idx = match.queryIdx
        x, y = kp1[img1_idx].pt
        matched_region.append((int(x), int(y)))

    if matched_region:
        x_min, y_min = np.min(matched_region, axis=0)
        x_max, y_max = np.max(matched_region, axis=0)

        # Increase the height at the top by subtracting from y_min
        y_min = max(0, y_min - height_increase)  # Ensuring y_min doesn't go below 0

        cv2.rectangle(group_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return (x_min, y_min, x_max, y_max)

# Function to detect faces using MTCNN with upward adjustment of the region on the original image
def detect_faces(image, region, max_attempts=5):
    detector = MTCNN()
    x_min, y_min, x_max, y_max = region
    face_images = []
    face_coords = []

    # Calculate the height of the rectangle
    height = y_max - y_min

    # Attempt to detect faces with upward adjustments if none are detected
    for attempt in range(max_attempts):
        # Create a copy of the original image to draw the rectangle
        temp_image = image.copy()

        # Draw the current search region on the image for visualization
        cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw rectangle in blue
        cv2.putText(temp_image, f'Attempt {attempt + 1}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 2)

        # Show the image with the current search region
        cv2.imshow(f'Attempt {attempt + 1}', temp_image)
        cv2.waitKey(500)  # Wait for 500ms to see each step (you can adjust this time)

        # Detect faces in the current region
        detected_faces = detector.detect_faces(image)

        for face in detected_faces:
            x, y, w, h = face['box']
            if x_min <= x <= x_max and y_min <= y <= y_max:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_images.append(image[y:y + h, x:x + w])  # Extract each face
                face_coords.append((x, y, w, h))

        # If faces are detected, close the attempt windows and return them
        if face_images:
            cv2.destroyAllWindows()  # Close all attempt windows
            return face_images, face_coords

        # If no faces are detected, adjust the rectangle:
        print(f"Attempt {attempt + 1}: No face detected, moving region upwards...")

        # Move the top edge down by height and set the bottom edge to the previous top edge
        y_min = max(0, y_min - height)  # Ensure y_min doesn't go below 0
        y_max = y_min + height  # Maintain the height

    # If no faces are detected after all attempts, return empty lists
    print("No faces detected after upward adjustments.")
    cv2.destroyAllWindows()  # Close all attempt windows
    return [], []

def case_2(group_image_path, individual_image_path):
    group_image, faces = detect_faces2(group_image_path)

    if faces:
        match_and_highlight_faces(group_image, faces, individual_image_path)
    else:
        print("No faces detected in the group photo.")

    # Show the result
    cv2.imshow('Group Photo with Highlighted Matches', group_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_and_highlight_faces(group_image, faces, individual_image_path):
    for face in faces:
        x, y, width, height = face['box']

        # Extract the detected face from the group image
        detected_face = group_image[y:y + height, x:x + width]

        # Save the detected face temporarily for DeepFace verification
        face_temp_path = 'detected_face.jpg'
        cv2.imwrite(face_temp_path, detected_face)

        # Use DeepFace to verify the detected face with the individual photo
        try:
            result = DeepFace.verify(face_temp_path, individual_image_path, enforce_detection=False)
            if result['verified']:
                # Highlight the matched face
                cv2.rectangle(group_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(group_image, "Matched", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error verifying face: {e}")


def detect_faces2(image_path):
    # Load the image
    image = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return image, faces

choice = int(input("Enter 1 for Face Detection with Pattern, 2 for Face Detection without Pattern: "))
if choice == 1:
    group_image_path = input("Enter path for group image: ")
    dress_image_path = input("Enter path for dress image: ")
    face_image_path = input("Enter path for face image: ")
    case_1(group_image_path, dress_image_path, face_image_path)
elif choice == 2:
    group_image_path = input("Enter path for group image: ")
    face_image_path = input("Enter path for face image: ")
    case_2(group_image_path, face_image_path)