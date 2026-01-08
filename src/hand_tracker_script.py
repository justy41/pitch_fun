import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Audio parameters
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
pitch_shift = 1.0  # 1.0 = normal pitch
audio_running = True

def audio_callback(indata, outdata, frames, time, status):
    """Process audio with pitch shifting"""
    global pitch_shift
    
    if status:
        print(status)
    
    # Simple pitch shifting using resampling
    if pitch_shift != 1.0:
        # Resample the audio
        indices = np.arange(0, len(indata), pitch_shift)
        indices = indices[indices < len(indata)].astype(int)
        shifted = indata[indices]
        
        # Resize to output length
        if len(shifted) < frames:
            shifted = np.pad(shifted, ((0, frames - len(shifted)), (0, 0)))
        else:
            shifted = shifted[:frames]
        
        outdata[:] = shifted
    else:
        outdata[:] = indata

def get_hand_height(hand_landmarks, image_height):
    """Get the average Y position of the hand (0 = top, 1 = bottom)"""
    y_positions = [lm.y for lm in hand_landmarks.landmark]
    return sum(y_positions) / len(y_positions)

# Start audio stream
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=2,
    callback=audio_callback
)
stream.start()

# Start video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("Controls:")
print("- Move hand UP = Higher pitch")
print("- Move hand DOWN = Lower pitch")
print("- Press 'q' to quit")

while True:
    success, image = cap.read()
    if not success:
        continue
    
    # Process image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    h, w, _ = image.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # Calculate hand height (0 to 1, where 0 is top)
            hand_y = get_hand_height(hand_landmarks, h)
            
            # Map hand position to pitch shift
            # Top (0) = 2.0x pitch (higher)
            # Bottom (1) = 0.5x pitch (lower)
            pitch_shift = 2.0 - (hand_y * 1.5)
            
            # Display pitch info
            pitch_text = f"Pitch: {pitch_shift:.2f}x"
            cv2.putText(image, pitch_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Reset to normal pitch when no hand detected
        pitch_shift = 1.0
        cv2.putText(image, "No hand detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Hand Pitch Control', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
hands.close()