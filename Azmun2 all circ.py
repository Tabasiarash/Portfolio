import cv2
import numpy as np
import json
import os
import time

# --- Configuration ---
TARGET_KEY_W = 800
TARGET_KEY_H = 1100
TOLERANCE_FACTOR = 0.03  # 3% tolerance for bubble matching
MIN_KEYPOINT_SIZE = 12   # Minimum size (in pixels) for ORB keypoints used for alignment

# --- Global State for Manual Selection and ROI ---
click_points = []
selection_mode = False
use_manual_warp = False
roi_box = None # Stores (x, y, w, h) of the crop area
last_contour = None # Stores the last successfully detected/manual paper outline

# --- File Paths ---
KEY_REF_PATH = "key_reference.jpg"
KEY_POS_PATH = "key_positions.json"
KEY_ROI_PATH = "key_roi.json" # New file path for the Region of Interest

def mouse_handler(event, x, y, flags, param):
    """Handles mouse clicks for manual corner selection."""
    global click_points
    global selection_mode
    global use_manual_warp

    if selection_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(click_points) < 4:
                click_points.append((x, y))
                print(f"Point {len(click_points)} captured at ({x}, {y})")
                
                # Draw point feedback on the live display
                temp_frame = param[0].copy()
                # If a key exists, draw its outline first
                if last_contour is not None:
                     cv2.drawContours(temp_frame, [last_contour], -1, (255, 255, 0), 5)
                     
                for i, (px, py) in enumerate(click_points):
                    cv2.circle(temp_frame, (px, py), 10, (0, 255, 255), -1)
                    cv2.putText(temp_frame, str(i+1), (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Live Camera Feed (Original)", temp_frame)

                if len(click_points) == 4:
                    print("✅ 4 points captured. Points are locked for the NEXT capture/grade operation.")
                    print("   - Press 'K' to use these points for the Key Sheet.")
                    print("   - Press 'R' to use these points for the Student Sheet grading.")
                    selection_mode = False
                    use_manual_warp = True
            else:
                print("Points already selected. Press 'P' to reset selection.")
                print("To use the current locked points, press 'K' (Key) or 'R' (Student Sheet).")


def order_and_warp(frame, points, target_width=TARGET_KEY_W, target_height=TARGET_KEY_H):
    """Orders 4 input points and warps the frame."""
    points = np.array(points, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    # Target points for the warped image
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (target_width, target_height))
    
    # Format for contour display
    contour = rect.astype(np.int32).reshape((-1, 1, 2))
    return warped, contour


def find_and_warp_paper(frame, target_width=TARGET_KEY_W, target_height=TARGET_KEY_H):
    """Detects the largest quadrilateral (assumed to be the paper) and warps it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paper_contour = None
    max_area = 0
    
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        
        if len(approx) == 4 and area > max_area:
            paper_contour = approx
            max_area = area

    if paper_contour is None:
        return None, None

    points = paper_contour.reshape(4, 2)
    return order_and_warp(frame, points, target_width, target_height)


def select_and_crop_roi(img):
    """
    Allows user to manually select a Region of Interest (ROI) using the mouse.
    Returns the cropped image and the (x, y, w, h) box.
    """
    # Temporarily set window properties to allow ROI selection
    cv2.namedWindow("Select ROI - Press ENTER or SPACE when done", cv2.WINDOW_AUTOSIZE)
    
    # Select ROI returns (x, y, w, h)
    roi = cv2.selectROI("Select ROI - Press ENTER or SPACE when done", img, False, False)
    cv2.destroyWindow("Select ROI - Press ENTER or SPACE when done")
    
    # Check if a valid area was selected
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        cropped_img = img[y:y+h, x:x+w]
        print(f"✅ ROI selected and applied: x={x}, y={y}, w={w}, h={h}")
        return cropped_img, roi
    else:
        print("⚠️ ROI selection cancelled or invalid. Using full image.")
        # Return the original image and a full-size ROI box
        h, w, _ = img.shape
        return img, (0, 0, w, h)


def detect_key_bubbles(warped_img):
    """Detects filled bubbles in the warped and cropped key image."""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # Dynamic thresholding logic
    mean_brightness = np.mean(gray)
    adaptive_threshold = max(60, min(140, mean_brightness * 0.8))

    def is_dark_dynamic(x, y, r):
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        return mean_intensity < adaptive_threshold

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=20, minRadius=8, maxRadius=25
    )
    if circles is None:
        return warped_img.copy(), {}

    circles = np.round(circles[0, :]).astype("int")
    
    key_positions = {}
    key_preview = warped_img.copy()

    for i, (x, y, r) in enumerate(circles):
        if is_dark_dynamic(x, y, r):
            # Mark filled answers in green
            cv2.rectangle(key_preview, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            # Store position as string index (for JSON serialization)
            key_positions[str(i)] = (float(x), float(y), float(r))
        else:
            # Mark empty circles in a light grey for context
            cv2.circle(key_preview, (x, y), r, (150, 150, 150), 1)

    return key_preview, key_positions


def align_paper_to_key(student_img, key_img):
    """Aligns the student image to the key image using ORB and Homography."""
    gray_student = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
    gray_key = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1_all, des1_all = orb.detectAndCompute(gray_student, None)
    kp2_all, des2_all = orb.detectAndCompute(gray_key, None)

    # Function to filter keypoints and their descriptors by size
    def filter_keypoints_and_descriptors(kp_all, des_all, min_size):
        if des_all is None:
            return [], None
        
        filtered_kp = []
        filtered_des_indices = []
        for i, kp in enumerate(kp_all):
            if kp.size >= min_size:
                filtered_kp.append(kp)
                filtered_des_indices.append(i)
        
        if not filtered_kp:
            return [], None
            
        filtered_des = des_all[filtered_des_indices]
        return filtered_kp, filtered_des

    kp1, des1 = filter_keypoints_and_descriptors(kp1_all, des1_all, MIN_KEYPOINT_SIZE)
    kp2, des2 = filter_keypoints_and_descriptors(kp2_all, des2_all, MIN_KEYPOINT_SIZE)

    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        print("⚠️ Failed to detect sufficient large ORB descriptors for robust alignment.")
        # Return a black image for the match visualization debug
        return student_img, np.eye(3), np.zeros((200, 200, 3), dtype=np.uint8)

    # Use Brute-Force Matcher on the filtered descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100] # Use top 100 matches

    if len(matches) < 15:
        print(f"⚠️ Only {len(matches)} keypoint matches found after size filtering. Alignment may be poor.")
        # Return a black image for the match visualization debug
        return student_img, np.eye(3), np.zeros((200, 200, 3), dtype=np.uint8)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("⚠️ Homography calculation failed.")
        return student_img, np.eye(3), np.zeros((200, 200, 3), dtype=np.uint8)

    aligned = cv2.warpPerspective(student_img, H, (key_img.shape[1], key_img.shape[0]))

    # Draw the matches for the debug window (will not be shown, but logic is kept)
    match_mask = mask.ravel().tolist()
    img_matches = cv2.drawMatches(
        student_img, kp1, key_img, kp2,
        matches, None,
        matchesMask=match_mask,
        singlePointColor=(0, 0, 255),
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return aligned, H, img_matches


def calculate_grade(aligned_student, key_positions):
    """Scores the aligned student sheet against the key positions."""
    # This logic operates on the already cropped and aligned image.
    gray = cv2.cvtColor(aligned_student, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    adaptive_threshold = max(60, min(140, mean_brightness * 0.8))

    def is_dark_dynamic(x, y, r):
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        return mean_intensity < adaptive_threshold

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=20, minRadius=8, maxRadius=25
    )
    if circles is None:
        return aligned_student.copy(), 0, 0

    circles = np.round(circles[0, :]).astype("int")

    # Filter circles to only include those that are filled (dark)
    student_filled = [(x, y, r) for (x, y, r) in circles if is_dark_dynamic(x, y, r)]
    
    graded = aligned_student.copy()
    paper_width = gray.shape[1]
    TOLERANCE = int(TOLERANCE_FACTOR * paper_width) 

    correct = 0
    matched_key_indices = set()

    parsed_key_positions = {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in key_positions.items()}
    
    for (sx, sy, sr) in student_filled:
        best_match = None
        min_dist = 1e9
        
        for idx, (kx, ky, kr) in parsed_key_positions.items():
            dx, dy = abs(kx - sx), abs(ky - sy)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            
            if dist < TOLERANCE and dist < min_dist:
                best_match = idx
                min_dist = dist

        if best_match is not None and best_match not in matched_key_indices:
            # Correct: Green box (Matches a key position)
            cv2.rectangle(graded, (sx - sr, sy - sr), (sx + sr, sy + sr), (0, 255, 0), 3)
            matched_key_indices.add(best_match)
            correct += 1
        else:
            # Wrong/Spurious: Red box
            cv2.rectangle(graded, (sx - sr, sy - sr), (sx + sr, sy + sr), (0, 0, 255), 3)
    
    # Mark missed answers (in key_positions but not matched by student)
    for idx, (kx, ky, kr) in parsed_key_positions.items():
        if idx not in matched_key_indices:
            # Missed: Yellow/Orange Box (shows where the key expected an answer)
            cv2.rectangle(graded, (kx - kr, ky - kr), (kx + kr, ky + kr), (0, 165, 255), 2)
            
    return graded, correct, len(student_filled)


# --- Orchestration Functions ---

def capture_key_from_camera(frame, manual_points=None):
    """Handles the multi-step key capture process (Warping only)."""
    global last_contour
    print("--- Capturing Key ---")
    
    # 1. Detect and Warp to A4 size
    if manual_points and len(manual_points) == 4:
        warped_key_a4, contour = order_and_warp(frame, manual_points)
    else:
        warped_key_a4, contour = find_and_warp_paper(frame)

    if warped_key_a4 is None:
        print("⚠️ No large paper contour detected (either automatically or from manual points).")
        return None, None, frame.copy()

    # 2. Save the full warped image (to be cropped later with 'C')
    cv2.imwrite(KEY_REF_PATH, warped_key_a4)

    # 3. Store the contour for persistent display
    last_contour = contour
    
    print("✅ Key paper captured. Press 'C' to define the bubble area (ROI), then 'K' again to process bubbles.")
    return warped_key_a4, contour


def process_key_with_roi(full_warped_key, key_positions, roi_box):
    """
    Applies the ROI crop to the key, re-runs bubble detection, and saves the final key data.
    """
    if roi_box:
        x, y, w, h = roi_box
        cropped_key = full_warped_key[y:y+h, x:x+w]
        print("Applying saved ROI to key sheet for bubble detection.")
    else:
        # If no ROI is defined, use the full image
        cropped_key = full_warped_key

    # 1. Bubble Detection on Cropped Image
    key_preview, new_key_positions = detect_key_bubbles(cropped_key)

    # 2. Save new key data
    cv2.imwrite(KEY_REF_PATH, cropped_key) # Overwrite reference with the cropped version
    with open(KEY_POS_PATH, "w") as f:
        json.dump(new_key_positions, f, indent=4)
    cv2.imwrite("key_filled_preview.jpg", key_preview)

    print(f"✅ Key processed with {len(new_key_positions)} filled bubbles. Using Cropped Image.")
    return cropped_key, new_key_positions, key_preview


def grade_student_sheet(student_img, key_img, key_positions, roi_box, manual_points=None):
    """Combines warping, alignment, and grading, applying the ROI crop."""
    
    # 1. Warp student image (Manual or Auto) to TARGET_KEY_W x TARGET_KEY_H
    if manual_points and len(manual_points) == 4:
        student_warp, _ = order_and_warp(student_img, manual_points)
    else:
        student_warp, _ = find_and_warp_paper(student_img)
            
    if student_warp is None:
        print("⚠️ Could not detect a clear paper outline on the student sheet.")
        return None, None, 0, 0, None, None
        
    # 2. Apply ROI Crop (This is the NEW cleaning step)
    if roi_box:
        x, y, w, h = roi_box
        # Ensure crop doesn't go out of bounds (though it shouldn't if ROI was set correctly on the key)
        student_cropped = student_warp[y:min(y+h, student_warp.shape[0]), x:min(x+w, student_warp.shape[1])]
    else:
        student_cropped = student_warp

    # Handle case where crop might result in an empty or tiny image
    if student_cropped.size == 0 or student_cropped.shape[0] < 50 or student_cropped.shape[1] < 50:
        print("⚠️ ROI crop resulted in an invalid/too small image. Alignment skipped.")
        return None, None, 0, 0, None, None
    
    # 3. Alignment & Match Visualization (img_matches is no longer used for display)
    aligned_student, _, _ = align_paper_to_key(student_cropped, key_img)
    
    if aligned_student is None:
        return None, None, 0, 0, None, None
    
    # 4. Grading on Aligned Sheet
    graded_overlay, correct, total_filled = calculate_grade(aligned_student, key_positions)

    return aligned_student, graded_overlay, correct, total_filled, None, student_warp


# --- Main Camera Loop and Interface ---

def run_grader():
    global click_points, selection_mode, use_manual_warp, roi_box, last_contour
    
    cap = cv2.VideoCapture(0)
    key_img = None
    key_positions = None
    full_warped_key_a4 = None # Used to store the uncropped key image after paper warp
    combined_result_to_save = None

    # We only want to see these two windows
    MAIN_WINDOWS = ["Live Camera Feed (Original)", "FINAL GRADE PREVIEW (Key | Graded Student)"]

    # Attempt to load saved data
    if os.path.exists(KEY_REF_PATH) and os.path.exists(KEY_POS_PATH):
        try:
            key_img = cv2.imread(KEY_REF_PATH)
            full_warped_key_a4 = key_img.copy() # Key_img is already the final cropped key if 'C' was used before
            with open(KEY_POS_PATH, "r") as f:
                key_positions = json.load(f)
            print(f"💾 Loaded existing key with {len(key_positions)} answers.")
            # Note: Not showing the key_img preview here, as per user request to hide debug views.
        except Exception as e:
            print(f"Error loading key files: {e}")
            key_img = None
            key_positions = None
            full_warped_key_a4 = None
    
    # Load ROI if it exists
    if os.path.exists(KEY_ROI_PATH):
        try:
            with open(KEY_ROI_PATH, "r") as f:
                roi_box = tuple(json.load(f))
            print(f"📐 Loaded saved ROI box: {roi_box}")
        except Exception as e:
            print(f"Error loading ROI file: {e}. Resetting ROI.")
            roi_box = None


    print("\n--- Auto Grader Initialized ---")

    # Define on-screen controls
    controls = [
        ("P", "Start/Reset Manual Warp Selection (Click 4 corners)"),
        ("K", "Capture Key Sheet (Warp only)"),
        ("C", "Define Crop Area (ROI) on Key Sheet"), 
        ("R", "Grade Current Sheet (Show Final Grade)"),
        ("S", "Save Last Combined Result"),
        ("Q", "Quit Application")
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.setMouseCallback("Live Camera Feed (Original)", mouse_handler, param=[frame])

        # Display on-screen controls and status
        display = frame.copy()
        
        # --- DRAW PERSISTENT CONTOUR IF KEY IS CAPTURED ---
        if key_img is not None and last_contour is not None:
            # Draw Cyan outline on the live feed as a guide
            cv2.drawContours(display, [last_contour], -1, (255, 255, 0), 5) 
        # --------------------------------------------------

        h, w, _ = display.shape
        cv2.putText(display, "BUBBLE SHEET GRADER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        status_text = "STATUS: READY - Press 'P' or 'K'" 
        status_color = (0, 165, 255)

        if selection_mode:
            status_text = f"MANUAL SELECTION: Click {len(click_points)+1} of 4 corners..."
            status_color = (0, 0, 255)
        elif use_manual_warp:
            status_text = f"WARP MODE: MANUAL (Points Locked). Press K or R."
            status_color = (255, 0, 255) # Magenta
        elif key_img is not None:
            roi_status = f"ROI: {roi_box[2]}x{roi_box[3]}" if roi_box else "ROI: None (Press 'C')"
            status_text = f"STATUS: Key Active ({len(key_positions)} ans) | {roi_status}"
            status_color = (0, 255, 0)
        
        cv2.putText(display, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        cv2.imshow("Live Camera Feed (Original)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            # Start manual selection mode
            click_points = []
            selection_mode = True
            use_manual_warp = False 

        elif key == ord('k'):
            # Key Capture Mode (Warp only)
            manual_warp_to_use = click_points if use_manual_warp and len(click_points) == 4 else None
            
            # The capture step now only performs the paper warp and stores the contour.
            new_full_warped, contour = capture_key_from_camera(frame, manual_warp_to_use)
            
            use_manual_warp = False 
            click_points = []

            if new_full_warped is not None:
                full_warped_key_a4 = new_full_warped # Store the uncropped key
                
                # If an ROI is already set, apply it and process bubbles immediately
                if roi_box:
                    key_img, key_positions, key_preview = process_key_with_roi(full_warped_key_a4, key_positions, roi_box)
                else:
                    # If no ROI is set, use the full A4 key as the active key_img for now
                    key_img = full_warped_key_a4.copy() 
                    print("Hint: Press 'C' to set the Region of Interest now.")

        elif key == ord('c'):
            # Define Crop Area (ROI)
            if full_warped_key_a4 is None:
                print("⚠️ Please capture the Key Sheet first (press 'K').")
            else:
                print("--- ROI Selection Started ---")
                
                # key_img will become the cropped image
                new_key_img, new_roi_box = select_and_crop_roi(full_warped_key_a4.copy()) 
                
                if new_roi_box:
                    roi_box = new_roi_box
                    key_img = new_key_img # Set the active key to the cropped version
                    
                    # Save the new ROI box coordinates
                    with open(KEY_ROI_PATH, "w") as f:
                        json.dump(list(roi_box), f, indent=4)
                    
                    # Process the key image with the new ROI
                    key_img, key_positions, key_preview = process_key_with_roi(full_warped_key_a4, key_positions, roi_box)


        elif key == ord('r'):
            # Grading Mode
            if key_img is None or key_positions is None:
                print("⚠️ Capture a key first (press 'K') and ensure it has bubbles (press 'C' if needed).")
                combined_result_to_save = None
            else:
                manual_points_to_use = click_points if use_manual_warp else None
                
                # Grade using the active key_img (which is cropped if 'C' was used) and the saved ROI box
                aligned_student, graded, correct, total_filled, _, _ = grade_student_sheet(
                    frame, key_img, key_positions, roi_box, manual_points_to_use
                )
                
                if graded is not None:
                    print(f"✅ Grading Result: {correct}/{len(key_positions)} correct. Student filled {total_filled} bubbles.")
                    
                    # --- SCORE TEXT ---
                    score_text = f"Score: {correct}/{len(key_positions)}"
                    score_y = 70 
                    score_font_scale = 1.0
                    score_font_thickness = 3
                    score_color = (0, 0, 0) # Black
                    cv2.putText(graded, score_text, (10, score_y), cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_color, score_font_thickness, cv2.LINE_AA)

                    # Stack Key with Graded Student
                    combined_img = np.hstack([key_img, graded])
                    
                    # --- ADD TITLES TO THE COMBINED IMAGE (Styling from prior request) ---
                    key_width = key_img.shape[1]
                    title_y = 30 # Position the title near the top
                    font_scale = 0.75 
                    font_thickness = 2
                    title_color = (0, 0, 0) # Black text
                    font_style = cv2.FONT_HERSHEY_SIMPLEX 

                    # Title for Key (Left Half)
                    cv2.putText(combined_img, "KEY", 
                                (key_width // 2 - 40, title_y), 
                                font_style, font_scale, title_color, font_thickness, cv2.LINE_AA)
                    
                    # Title for Student (Right Half)
                    cv2.putText(combined_img, "STUDENT", 
                                (key_width + key_width // 2 - 70, title_y), 
                                font_style, font_scale, title_color, font_thickness, cv2.LINE_AA)
                    # --- END TITLES ---
                    
                    combined_result_to_save = combined_img.copy()

                    cv2.imshow("FINAL GRADE PREVIEW (Key | Graded Student)", combined_result_to_save)
                    
                use_manual_warp = False
                click_points = []


        elif key == ord('s'):
            if combined_result_to_save is not None:
                filename = f"graded_result_{int(time.time())}.jpg"
                cv2.imwrite(filename, combined_result_to_save)
                print(f"💾 Saved combined result (Key and Graded Student) as {filename}")
            else:
                print("⚠️ Please grade a sheet first (press 'R') before saving.")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_grader()
