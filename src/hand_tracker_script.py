import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy import signal

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Manual device configuration - based on your system
CABLE_INPUT_DEVICE = 2   # CABLE Output (VB-Audio Virtual Cable)
SPEAKER_OUTPUT_DEVICE = 6  # Speaker (2- Realtek(R) Audio)

# Audio parameters
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
pitch_shift_factor = 1.0
audio_buffer = np.zeros((BLOCK_SIZE * 4, 2))
buffer_index = 0

def pitch_shift_audio(audio, factor):
    """Pitch shift audio using phase vocoder technique"""
    if factor == 1.0:
        return audio
    
    # Simple time-stretch and resample method
    stretched = signal.resample(audio, int(len(audio) / factor))
    
    # Pad or truncate to match input length
    if len(stretched) > len(audio):
        return stretched[:len(audio)]
    else:
        padded = np.zeros_like(audio)
        padded[:len(stretched)] = stretched
        return padded

def audio_callback(indata, outdata, frames, time, status):
    """Capture system audio and output pitch-shifted version"""
    global pitch_shift_factor
    
    if status:
        print(status)
    
    try:
        # Apply pitch shifting
        if pitch_shift_factor != 1.0:
            shifted = pitch_shift_audio(indata.copy(), pitch_shift_factor)
            outdata[:] = shifted
        else:
            outdata[:] = indata
    except Exception as e:
        print(f"Audio error: {e}")
        outdata[:] = indata

def get_hand_height(hand_landmarks):
    """Get the average Y position of the hand (0 = top, 1 = bottom)"""
    y_positions = [lm.y for lm in hand_landmarks.landmark]
    return sum(y_positions) / len(y_positions)

# Display available audio devices
print("\nAvailable audio devices:")
print(sd.query_devices())

print(f"\nUsing:")
print(f"Input: Device {CABLE_INPUT_DEVICE}")
print(f"Output: Device {SPEAKER_OUTPUT_DEVICE}")

try:
    # Start audio stream with manual device configuration
    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=(CABLE_INPUT_DEVICE, SPEAKER_OUTPUT_DEVICE),
        channels=2,
        callback=audio_callback,
        dtype='float32'
    )
    stream.start()
    
except Exception as e:
    print(f"\nError setting up audio: {e}")
    print("\nMake sure:")
    print("1. VB-Audio Virtual Cable is installed")
    print("2. Windows default playback device is set to 'CABLE Input'")
    exit()

# Start video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("\n" + "="*50)
print("HAND PITCH CONTROL - SYSTEM AUDIO")
print("="*50)
print("Controls:")
print("- Move hand UP = Higher pitch (chipmunk effect)")
print("- Move hand DOWN = Lower pitch (slow-mo effect)")
print("- Press 'q' to quit")
print("="*50 + "\n")

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
            hand_y = get_hand_height(hand_landmarks)
            
            # Map hand position to pitch shift
            # Top (0) = 2.0x pitch (higher/faster)
            # Middle (0.5) = 1.0x (normal)
            # Bottom (1) = 0.5x pitch (lower/slower)
            pitch_shift_factor = 2.0 - (hand_y * 1.5)
            
            # Clamp values
            pitch_shift_factor = max(0.5, min(2.0, pitch_shift_factor))
            
            # Display info
            pitch_text = f"Pitch: {pitch_shift_factor:.2f}x"
            if pitch_shift_factor > 1.2:
                effect = "HIGHER"
                color = (0, 255, 255)
            elif pitch_shift_factor < 0.8:
                effect = "LOWER"
                color = (255, 0, 255)
            else:
                effect = "NORMAL"
                color = (0, 255, 0)
            
            cv2.putText(image, pitch_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, effect, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # Reset to normal pitch when no hand detected
        pitch_shift_factor = 1.0
        cv2.putText(image, "No hand - Normal pitch", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw guide lines
    cv2.line(image, (0, h//4), (w, h//4), (0, 255, 255), 1)  # High pitch line
    cv2.line(image, (0, h//2), (w, h//2), (0, 255, 0), 1)    # Normal pitch line
    cv2.line(image, (0, 3*h//4), (w, 3*h//4), (255, 0, 255), 1)  # Low pitch line
    
    cv2.imshow('Hand Pitch Control - System Audio', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\nCleaning up...")
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Don't forget to change your default audio output back to your speakers!")