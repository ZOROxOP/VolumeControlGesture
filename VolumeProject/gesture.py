import math 
from ctypes import POINTER , cast
import cv2
import importlib

try:
  mp = importlib.import_module("mediapipe")
except ModuleNotFoundError as exc:
  raise RuntimeError(
    "MediaPipe is required. Install it with: python -m pip install mediapipe"
  ) from exc
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume

CAM_INDEX = 0

FRAME_W,FRAME_H = 1280,720
MIN_DIST = 30
MAX_DIST = 200
SMOOTHING = 0.25  #USE TO INCREASE speed of response 

THUMB_TIP,INDEX_TIP = 4,8
FINGER_TIPS = (8,12,16,20)
FINGER_MCPS = (5,9,13,17)
PINKY_TIP,PINKY_PIP = 20,18

GREEN,RED,YELLOW,WHITE,GRAY = (0,220,0),(0,0,230),(0,220,220),(255,255,255),(120,120,120)

def get_volume_interface():  #finds the default audio device 
  """Return the Windows master -volume endpoint(works across pycaw versions)"""
  speakers = AudioUtilities.GetSpeakers()
  if hasattr(speakers, "EndpointVOlume"):
    return speakers.EndpointVolume
  interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
  return cast(interface, POINTER(IAudioEndpointVolume))

def pixel_coords(landmark,w,h):
  """Convert normalized landmark coordinates to pixel coordinates"""
  return int(landmark.x * w), int(landmark.y * h)

def is_pinky_raised(lm):
  """Check if the pinky finger is raised"""
  return lm[PINKY_TIP].y < lm[PINKY_PIP].y

def is_fist(lm):
    """Check if the hand is in a fist position"""
    return all(lm[tip].y > lm[mcp].y for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS))

def draw_hud(img,vol_percent,state,is_muted):
  color = {"CONTROLLING": GREEN,"LOCKED":YELLOW,"MUTED":RED}.get(state,GRAY)
  x1, y1, x2, y2 = 50, 150, 85, 400
  fill_y = int(np.interp(vol_percent, [0, 100], [y2, y1]))
  cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 2)
  cv2.rectangle(img, (x1, fill_y), (x2, y2), RED if is_muted else GREEN, cv2.FILLED)
  cv2.putText(img, f"{int(vol_percent)}%", (x1 - 10, y2 + 40),
              cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

  cv2.putText(img, f"STATE: {state}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
  cv2.putText(img, "Pinky up + pinch = volume | Fist = mute | q = quit",
              (30, FRAME_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

  def main():
    volume = get_volume_interface()
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
      raise RuntimeError("Could not open webcam. Please check camera index or connection.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
      static_image_mode=False,
      max_num_hands=1,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7,
    )
    smoothed_vol = volume.GetMasterVolumeLevelScalar()
    muted_by_app = False

    try:
      while True:
        ok, frame = cap.read()
        if not ok:
          break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        state = "LOCKED"
        if results.multi_hand_landmarks:
          hand = results.multi_hand_landmarks[0]
          lm = hand.landmark
          mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

          if is_fist(lm):
            state = "MUTED"
            if not muted_by_app:
              volume.SetMute(1, None)
              muted_by_app = True
          else:
            if muted_by_app:
              volume.SetMute(0, None)
              muted_by_app = False

            x1, y1 = pixel_coords(lm[THUMB_TIP], w, h)
            x2, y2 = pixel_coords(lm[INDEX_TIP], w, h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = math.hypot(x2 - x1, y2 - y1)
            controlling = is_pinky_raised(lm)
            line_color = GREEN if controlling else YELLOW
            cv2.circle(frame, (x1, y1), 10, line_color, cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, line_color, cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)
            cv2.circle(frame, (cx, cy), 8,
                   RED if dist < MIN_DIST else line_color, cv2.FILLED)

            if controlling:
              state = "CONTROLLING"
              target = float(np.interp(dist, [MIN_DIST, MAX_DIST], [0.0, 1.0]))
              smoothed_vol += SMOOTHING * (target - smoothed_vol)
              volume.SetMasterVolumeLevelScalar(
                float(np.clip(smoothed_vol, 0, 1)), None
              )

        current_vol = volume.GetMasterVolumeLevelScalar() * 100
        draw_hud(frame, current_vol, state, bool(volume.GetMute()))
        cv2.imshow("Gesture Volume Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break

    finally:
      if muted_by_app:
        volume.SetMute(0, None)
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

        if __name__ == "__main__":
          main()