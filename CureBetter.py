
# CURE BETTER


import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
import traceback
 
os.makedirs("snapshots", exist_ok=True)

# ------------- CONFIG -------------
BASELINE_FRAMES = 90           
SMOOTH_ALPHA = 0.22            
PROJ_FACTOR = 0.62
NUM_FOREHEAD_SAMPLES = 13      
NUM_BROW_EXTRA = 6             
HIGH_MIN = 80.0
MODERATE_MIN = 55.0


W_BROW = 0.55
W_EYE = 0.38
W_MOUTH = 0.07   

# z->feature sensitivity
Z_DIVISOR = 1.4

MIN_RAW_TO_ZERO = 4.0          
# landmark indices we will use 
L_BROW_IN, R_BROW_IN = 70, 300
FOREHEAD_TOP, CHIN = 10, 152
# eyes
L_EYE_TOP, L_EYE_BOTTOM, L_EYE_LEFT, L_EYE_RIGHT = 159, 145, 33, 133
R_EYE_TOP, R_EYE_BOTTOM, R_EYE_LEFT, R_EYE_RIGHT = 386, 374, 362, 263
# mouth
TOP_LIP, BOTTOM_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

# cheek-ish extra points (approx medial cheek line)
CHEEK_LEFT_IDX = 234   # approximate; safe to use
CHEEK_RIGHT_IDX = 454

# ------------- helpers -------------
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def px_safe(lm, idx, W, H):
    if idx < 0 or idx >= len(lm):
        return None
    p = lm[idx]
    return (int(p.x * W), int(p.y * H))

def z_to_feat(z):
    if z <= 0:
        return 0.0
    v = math.tanh(min(z / Z_DIVISOR, 3.0))
    return float(np.clip(v, 0.0, 1.0))

def inflate_std(std, mean):
    return max(std, abs(mean) * 0.04, 1e-4)

# ------------- mediapipe -------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ------------- state -------------
baseline = None
capturing_baseline = False
baseline_buf = []
smoothed_score = 0.0
last_snapshot = 0.0

print("more_dots_improved_accuracy.py (final tuned)")
print(" - Press 'b' to capture baseline (neutral, no smile/frown). Hold still until 'Baseline saved'.")
print(" - After baseline: neutral=None; smile=Mild; squint/frown -> Moderate; strong grimace -> High. ESC to quit.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # MediaPipe
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
        except Exception as e:
            res = None
            print("MediaPipe error:", e)

        metrics = None
        if res and res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # extract key safe points
            idxs = [L_BROW_IN, R_BROW_IN, FOREHEAD_TOP, CHIN,
                    L_EYE_TOP, L_EYE_BOTTOM, R_EYE_TOP, R_EYE_BOTTOM,
                    TOP_LIP, BOTTOM_LIP, MOUTH_LEFT, MOUTH_RIGHT,
                    CHEEK_LEFT_IDX, CHEEK_RIGHT_IDX]
            okpts = True
            pts = {}
            for i in idxs:
                p = px_safe(lm, i, W, H)
                if p is None:
                    okpts = False
                    break
                pts[i] = p
            if okpts:
                face_h = max(1.0, dist(pts[FOREHEAD_TOP], pts[CHIN]))

                # brows & forehead
                left_brow = pts[L_BROW_IN]; right_brow = pts[R_BROW_IN]
                inner_brow_gap = dist(left_brow, right_brow) / face_h
                brow_to_eye = (dist(left_brow, pts[L_EYE_TOP]) + dist(right_brow, pts[R_EYE_TOP])) / 2.0 / face_h

                # generate many forehead samples across medial region
                forehead_samples_px = []
                forehead_ys_norm = []
                for i in range(NUM_FOREHEAD_SAMPLES):
                    t = i / (NUM_FOREHEAD_SAMPLES - 1) if NUM_FOREHEAD_SAMPLES > 1 else 0.5
                    bx = int(left_brow[0] + (right_brow[0] - left_brow[0]) * t)
                    by = int(left_brow[1] + (right_brow[1] - left_brow[1]) * t)
                    fx = int(bx + PROJ_FACTOR * (pts[FOREHEAD_TOP][0] - bx))
                    fy = int(by + PROJ_FACTOR * (pts[FOREHEAD_TOP][1] - by))
                    forehead_samples_px.append((fx, fy))
                    forehead_ys_norm.append(fy / face_h)

                # eyes
                left_eye = dist(pts[L_EYE_TOP], pts[L_EYE_BOTTOM]) / face_h
                right_eye = dist(pts[R_EYE_TOP], pts[R_EYE_BOTTOM]) / face_h
                eye_norm = (left_eye + right_eye) / 2.0

                # mouth
                mouth_open = dist(pts[TOP_LIP], pts[BOTTOM_LIP]) / face_h
                mouth_width = dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / face_h
                left_corner_y = pts[MOUTH_LEFT][1] / face_h
                right_corner_y = pts[MOUTH_RIGHT][1] / face_h

                # cheeks (extra dots for visibility)
                cheek_left = pts[CHEEK_LEFT_IDX]; cheek_right = pts[CHEEK_RIGHT_IDX]

                metrics = {
                    "face_h": face_h,
                    "inner_brow_gap": inner_brow_gap,
                    "brow_to_eye": brow_to_eye,
                    "forehead_ys": forehead_ys_norm,
                    "forehead_px": forehead_samples_px,
                    "eye_norm": eye_norm,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "mouth_open": mouth_open,
                    "mouth_width": mouth_width,
                    "left_corner_y": left_corner_y,
                    "right_corner_y": right_corner_y,
                    "cheek_left": cheek_left,
                    "cheek_right": cheek_right,
                    "pts": pts
                }

        # baseline capture flow
        if capturing_baseline:
            if metrics is not None:
                baseline_buf.append(metrics)
            cv2.putText(frame, f"Capturing baseline {len(baseline_buf)}/{BASELINE_FRAMES}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
            if len(baseline_buf) >= BASELINE_FRAMES:
                # compute baseline mean/std for each scalar metric
                baseline = {}
                # numeric scalar keys (added corner y means for smile detection)
                scalar_keys = ["inner_brow_gap", "brow_to_eye", "eye_norm", "left_eye", "right_eye", "mouth_open", "mouth_width",
                               "left_corner_y", "right_corner_y"]
                for k in scalar_keys:
                    vals = np.array([b[k] for b in baseline_buf], dtype=float)
                    baseline[f"{k}_mean"] = float(np.mean(vals))
                    baseline[f"{k}_std"] = float(np.std(vals) + 1e-6)
                # forehead: per-sample mean/std
                arr = np.array([b["forehead_ys"] for b in baseline_buf])  # shape (frames, samples)
                baseline["forehead_ys_mean"] = np.mean(arr, axis=0).tolist()
                baseline["forehead_ys_std"] = np.std(arr, axis=0).tolist()
                baseline["face_h_mean"] = float(np.mean([b["face_h"] for b in baseline_buf]))
                # save a representative pts mapping for drawing baseline dots if needed
                baseline["example_pts"] = baseline_buf[0]["pts"]
                baseline_buf = []
                capturing_baseline = False
                print("Baseline saved. You are ready to test expressions.")
        else:
            if baseline is None:
                cv2.putText(frame, "Press 'b' to capture baseline (neutral face, no smile/frown)", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # compute score when baseline & metrics exist
        raw_score = 0.0
        smile_flag = False
        if baseline is not None and metrics is not None:
            try:
                # brow z-scores (we use decrease: baseline_mean - current)/std
                mean_gap = baseline["inner_brow_gap_mean"]; std_gap = inflate_std(baseline["inner_brow_gap_std"], mean_gap)
                z_gap = (mean_gap - metrics["inner_brow_gap"]) / std_gap
                mean_bte = baseline["brow_to_eye_mean"]; std_bte = inflate_std(baseline["brow_to_eye_std"], mean_bte)
                z_bte = (mean_bte - metrics["brow_to_eye"]) / std_bte
                gap_feat = z_to_feat(z_gap)
                bte_feat = z_to_feat(z_bte)

                # forehead: standardize mean y across samples
                base_ys_mean = np.array(baseline.get("forehead_ys_mean", []))
                if base_ys_mean.size:
                    cur_mean_y = float(np.mean(metrics["forehead_ys"]))
                    base_mean_y = float(np.mean(base_ys_mean))
                    base_std_y = float(np.mean(baseline.get("forehead_ys_std", [0.0]*len(base_ys_mean))) + 1e-6)
                    z_fore = (cur_mean_y - base_mean_y) / max(base_std_y, abs(base_mean_y)*0.03, 1e-4)
                    forehead_feat = z_to_feat(z_fore)
                else:
                    forehead_feat = 0.0

                brow_feat = float(np.clip(0.57*gap_feat + 0.28*bte_feat + 0.15*forehead_feat, 0.0, 1.0))

                # eyes
                mean_eye = baseline["eye_norm_mean"]; std_eye = inflate_std(baseline["eye_norm_std"], mean_eye)
                z_eye = (mean_eye - metrics["eye_norm"]) / std_eye
                eye_feat = z_to_feat(z_eye)

                # mouth tension (pressed) - small effect now
                mean_mo = baseline["mouth_open_mean"]; std_mo = inflate_std(baseline["mouth_open_std"], mean_mo)
                z_mt1 = (mean_mo - metrics["mouth_open"]) / std_mo
                mean_mw = baseline["mouth_width_mean"]; std_mw = inflate_std(baseline["mouth_width_std"], mean_mw)
                z_mt2 = (mean_mw - metrics["mouth_width"]) / std_mw
                mouth_tension_feat = float(np.clip(0.6*z_to_feat(z_mt1) + 0.4*z_to_feat(z_mt2), 0.0, 1.0))

                # mouth_open high detection (only big mouth open -> High)
                z_mo_inc = (metrics["mouth_open"] - mean_mo) / std_mo
                mouth_open_feat = z_to_feat(z_mo_inc)

                # smile detection: width increased significantly AND corner y decreased (corners raised)
                width_mean = baseline["mouth_width_mean"]
                corners_base_mean = (baseline.get("left_corner_y_mean", metrics["left_corner_y"]) + baseline.get("right_corner_y_mean", metrics["right_corner_y"])) / 2.0
                corner_cur_mean = (metrics["left_corner_y"] + metrics["right_corner_y"]) / 2.0
                smile_flag = (metrics["mouth_width"] > width_mean * 1.18) and (corner_cur_mean < corners_base_mean - 0.01)

                # Compose base combined (brow + eye dominate; mouth tension small)
                base_combined = (W_BROW * brow_feat) + (W_EYE * eye_feat) + (W_MOUTH * mouth_tension_feat)
                raw_score = float(np.clip(base_combined * 100.0, 0.0, 100.0))

                # explicit Moderate rule: modest eye squeeze + brow furrow -> Moderate
                if (eye_feat >= 0.22 and brow_feat >= 0.22) or (eye_feat >= 0.18 and brow_feat >= 0.28):
                    raw_score = max(raw_score, 60.0)

                # explicit High rules:
                # - strong eye squeeze (squint almost closed) OR
                # - deep brow furrow OR
                # - big mouth open (sustained)
                if (eye_feat >= 0.48 and brow_feat >= 0.30) or (brow_feat >= 0.50) or (mouth_open_feat >= 0.65) or (metrics["mouth_open"] > 0.30):
                    raw_score = max(raw_score, 88.0)

                # smile suppression: if smile detected, strongly reduce score
                if smile_flag:
                    raw_score = min(raw_score, 28.0)

                # eliminate tiny noise
                if raw_score < MIN_RAW_TO_ZERO:
                    raw_score = 0.0

                # smoothing
                smoothed_score = (SMOOTH_ALPHA * raw_score) + ((1.0 - SMOOTH_ALPHA) * smoothed_score)

            except Exception as e:
                print("Feature calc error:", e)
                traceback.print_exc()
                smoothed_score = smoothed_score  # keep old

        # draw more dots: forehead samples + extra medial brow dots + cheeks
        if res and res.multi_face_landmarks and metrics is not None:
            lm = res.multi_face_landmarks[0].landmark
            pts = metrics["pts"]
            lb = pts[L_BROW_IN]; rb = pts[R_BROW_IN]
            # draw extra brow medial dots
            for i in range(NUM_BROW_EXTRA + 1):
                t = i / (NUM_BROW_EXTRA)
                bx = int(lb[0] + (rb[0] - lb[0]) * t)
                by = int(lb[1] + (rb[1] - lb[1]) * t)
                cv2.circle(frame, (bx, by), 3, (95, 180, 255), -1)
            # forehead samples
            for p in metrics["forehead_px"]:
                cv2.circle(frame, p, 3, (200, 100, 200), -1)
            # cheeks
            if metrics.get("cheek_left"):
                cv2.circle(frame, metrics["cheek_left"], 3, (180,150,80), -1)
            if metrics.get("cheek_right"):
                cv2.circle(frame, metrics["cheek_right"], 3, (180,150,80), -1)
            # eyes & lips
            lt = px_safe(lm, L_EYE_TOP, W, H); lbp = px_safe(lm, L_EYE_BOTTOM, W, H)
            rt = px_safe(lm, R_EYE_TOP, W, H); rbp = px_safe(lm, R_EYE_BOTTOM, W, H)
            if lt: cv2.circle(frame, lt, 3, (0,255,0), -1)
            if lbp: cv2.circle(frame, lbp, 3, (0,255,0), -1)
            if rt: cv2.circle(frame, rt, 3, (0,255,0), -1)
            if rbp: cv2.circle(frame, rbp, 3, (0,255,0), -1)
            tl = px_safe(lm, TOP_LIP, W, H); bl = px_safe(lm, BOTTOM_LIP, W, H)
            ml = px_safe(lm, MOUTH_LEFT, W, H); mr = px_safe(lm, MOUTH_RIGHT, W, H)
            if tl: cv2.circle(frame, tl, 3, (0,0,255), -1)
            if bl: cv2.circle(frame, bl, 3, (0,0,255), -1)
            if ml: cv2.circle(frame, ml, 3, (255,200,0), -1)
            if mr: cv2.circle(frame, mr, 3, (255,200,0), -1)

        # overlays: baseline instruction or score
        if capturing_baseline:
            cv2.putText(frame, f"Capturing baseline {len(baseline_buf)}/{BASELINE_FRAMES}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
        elif baseline is None:
            cv2.putText(frame, "Press 'b' to capture baseline (neutral face, no smile/frown)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        else:
            band = "None"; col = (0,200,0)
            if smoothed_score >= HIGH_MIN:
                band = "High"; col = (0,0,255)
            elif smoothed_score >= MODERATE_MIN:
                band = "Moderate"; col = (0,140,255)
            elif smoothed_score >= 11:
                band = "Mild"; col = (0,200,255)
            cv2.putText(frame, f"Pain Score: {smoothed_score:.1f}", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
            cv2.putText(frame, f"Band: {band}", (10, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

        cv2.imshow("cure better", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('b') and not capturing_baseline:
            # start baseline capture
            capturing_baseline = True
            baseline_buf = []
            print("Started baseline capture: hold neutral (no smile/frown) until done.")
        # save snapshot when High
        now = time.time()
        if baseline is not None and smoothed_score >= HIGH_MIN and now - last_snapshot > 5.0:
            fname = os.path.join("snapshots", f"high_{int(now)}.jpg")
            cv2.imwrite(fname, frame)
            last_snapshot = now
            print("Saved snapshot:", fname, "score:", smoothed_score)

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Exited.")
