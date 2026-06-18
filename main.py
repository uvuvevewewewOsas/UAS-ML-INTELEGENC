import pygame as pg
import pygame
import tkinter as tk
from tkinter import filedialog
import pygame.display
import threading
import time
import sys
import os
import os.path
import librosa
import numpy as np
import pandas as pd
import random
import joblib
import cv2
import mediapipe as mp
import heapq
from collections import deque
from scipy.fftpack import dct
from scipy.interpolate import interp1d
import libfmp.b
import libfmp.c3
import libfmp.c8

# --- Global Configurations ---
TITLE = "Melody's Quest"
WIDTH = 512
HEIGHT = 512
HEIGHT2 = 256.5  # height of a bar
WIDTH2 = 88      # width of a bar
FPS = 60         # frame rate

# Lane coordinates and mappings
# Lebar layar 512 dibagi 4 = 128 piksel per jalur
LANE_X = [0, 128, 256, 384] 
key_list = [40, 167, 297, 417]
X_TO_LANE = {40: 0, 167: 1, 297: 2, 417: 3}
LANE_TO_X = {0: 0, 1: 128, 2: 256, 3: 384}

# Resolve absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_asset_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- AI Agents & CV Helpers ---

class CameraTracker:
    """
    Manages OpenCV camera frames and Mediapipe hand tracking in a background thread
    to prevent blocking Pygame's main rendering thread (maintaining 60 FPS).
    """
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None  # RGB image for Pygame background
        self.finger_x = None      # Mapped to screen space (0 to width)
        self.finger_y = None      # Mapped to screen space (0 to height)
        self.finger_x2 = None     # Second finger for dual-hand support
        self.finger_y2 = None     # Second finger for dual-hand support
        self.overlay_enabled = True  # Auto-enable camera on startup
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _run(self):
        # Coba buka kamera dengan CAP_DSHOW, jika gagal coba tanpa itu
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
        if not self.cap.isOpened():
            print("EROR: Webcam tidak terdeteksi! Pastikan tidak sedang digunakan aplikasi lain.")
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Changed to detect up to two hands
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        mp_draw = mp.solutions.drawing_utils

        print("Kamera berhasil terhubung. Memulai pelacakan tangan...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror and resize frame to screen dimensions
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))

            # Convert BGR to RGB for Pygame and Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            finger_x, finger_y = None, None
            finger_x2, finger_y2 = None, None
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Index finger tip is landmark #8
                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    fx = int(tip.x * self.width)
                    fy = int(tip.y * self.height)

                    # Draw landmarks on the RGB frame
                    mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.circle(rgb_frame, (fx, fy), 8, (0, 255, 0), -1)

                    # Store first hand in finger_x/y, second hand in finger_x2/y2
                    if idx == 0:
                        finger_x, finger_y = fx, fy
                    elif idx == 1:
                        finger_x2, finger_y2 = fx, fy

            with self.lock:
                self.finger_x = finger_x
                self.finger_y = finger_y
                self.finger_x2 = finger_x2
                self.finger_y2 = finger_y2
                self.latest_frame = rgb_frame

        self.cap.release()


class UninformedSearchAgent:
    """
    Auto-Play Bot using Breadth-First Search (BFS) to find the shortest path
    of lane transitions to catch all active notes in the hit area.
    """
    def __init__(self):
        pass

    def find_path(self, start_lane, target_lanes):
        """
        Finds the shortest sequence of lane transitions to hit all targeted lanes.
        """
        if not target_lanes:
            return []

        # Queue elements: (current_lane, path_list, remaining_targets_set)
        queue = deque([(start_lane, [], set(target_lanes))])
        visited = set()

        while queue:
            curr_lane, path, remaining = queue.popleft()

            state_key = (curr_lane, tuple(sorted(list(remaining))))
            if state_key in visited:
                continue
            visited.add(state_key)

            if not remaining:
                return path

            # Generate transitions to all lanes
            for next_lane in range(4):
                if next_lane == curr_lane:
                    continue

                new_remaining = remaining.copy()
                if next_lane in new_remaining:
                    new_remaining.remove(next_lane)

                queue.append((next_lane, path + [next_lane], new_remaining))

        return []


class InformedSearchAgent:
    """
    A* Search spawner that dynamically plans a path for spawning "bonus notes"
    based on predicted music arousal and valence (energy and mood).
    """
    def __init__(self, arousal=0.0, valence=0.0):
        self.arousal = max(-1.0, min(1.0, float(arousal)))
        self.valence = max(-1.0, min(1.0, float(valence)))

    def find_bonus_path(self, start_lane, player_lane, path_length=5):
        """
        Uses A* to find an optimal sequence of lanes for spawning bonus notes.
        Heuristic: Manhattan distance to the player's current hand/cursor position.
        """
        # Heuristic: Manhattan distance to player's lane, scaled by remaining steps.
        # Guides the path to end near the player's recent hand/indicator position.
        def heuristic(lane, step):
            return abs(lane - player_lane) * (path_length - step)

        # Heap contains: (f_score, g_score, current_lane, step, path)
        start_h = heuristic(start_lane, 0)
        open_set = [(start_h, 0.0, start_lane, 0, [])]

        while open_set:
            f, g, lane, step, path = heapq.heappop(open_set)

            if step == path_length:
                return path

            for next_lane in range(4):
                dist = abs(next_lane - lane)

                # Arousal factor: high energy -> smaller transition penalty -> allows larger leaps
                arousal_factor = 1.0 / (1.0 + max(0.0, self.arousal))
                step_cost = dist * arousal_factor

                # Valence influence: high valence rewards smooth transitions (+-1)
                valence_bonus = 0.0
                if dist == 1:
                    valence_bonus = -0.3 * max(0.0, self.valence)

                # Negative valence increases penalty for large transitions
                if self.valence < 0 and dist > 1:
                    valence_bonus = 0.2 * abs(self.valence)

                g_next = g + step_cost + valence_bonus + 0.1
                h_next = heuristic(next_lane, step + 1)
                f_next = g_next + h_next

                heapq.heappush(open_set, (f_next, g_next, next_lane, step + 1, path + [next_lane]))

        return [start_lane] * path_length


class FuzzyDifficultyController:
    """
    Fuzzy Logic System to determine dynamic game speed.
    Inputs: Combo (performance), Arousal (music energy).
    Output: Game Speed.
    """
    def __init__(self):
        pass

    def compute_speed(self, current_combo, arousal):
        # Simple Fuzzy Membership & Inference
        # Combo: Low (0-20), High (>20)
        # Arousal: Chill (<0), Energetic (>0)
        
        mu_combo_high = min(1.0, current_combo / 20.0)
        mu_combo_low = 1.0 - mu_combo_high
        
        mu_arousal_high = max(0.0, min(1.0, (arousal + 1) / 2))
        mu_arousal_low = 1.0 - mu_arousal_high

        # Rules:
        # 1. If Combo is High AND Arousal is Energetic -> Fast
        # 2. If Combo is Low AND Arousal is Chill -> Slow
        # 3. Otherwise -> Medium
        
        rule1 = min(mu_combo_high, mu_arousal_high) # Weight for Fast
        rule2 = min(mu_combo_low, mu_arousal_low)   # Weight for Slow
        rule3 = 1.0 - max(rule1, rule2)             # Weight for Medium
        
        # Defuzzification (Weighted Average)
        speed_slow = 4.0
        speed_med = 6.0
        speed_fast = 9.0
        
        calculated_speed = (rule1 * speed_fast + rule2 * speed_slow + rule3 * speed_med) / (rule1 + rule2 + rule3 + 1e-6)
        return calculated_speed


# --- Original Utility Functions ---

def separate_melody_accompaniment(x, Fs, N, H, traj, n_harmonics=10, tol_cent=50.0):
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, pad_mode='constant')
    Fs_feature = Fs / H
    T_coef = np.arange(X.shape[1]) / Fs_feature
    freq_res = Fs / N
    F_coef = np.arange(X.shape[0]) * freq_res

    traj_X_values = interp1d(traj[:, 0], traj[:, 1], kind='nearest', fill_value='extrapolate')(T_coef)
    traj_X = np.hstack((T_coef[:, None], traj_X_values[:, None]))

    mask_mel = libfmp.c8.convert_trajectory_to_mask_cent(traj_X, F_coef, n_harmonics=n_harmonics, tol_cent=tol_cent)
    mask_acc = np.ones(mask_mel.shape) - mask_mel

    X_mel = X * mask_mel
    X_acc = X * mask_acc

    x_mel = librosa.istft(X_mel, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_acc = librosa.istft(X_acc, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    return x_mel, x_acc


def random_discard(onset_times, onset_frames, threshold=1):
    i = 1
    new_onset_times = [onset_times[0]]
    new_onset_frames = [onset_frames[0]]
    while i < len(onset_times):
        diff = onset_times[i] - onset_times[i - 1]
        if not diff < threshold or not random.random() >= 0.5:
            new_onset_times.append(onset_times[i])
            new_onset_frames.append(onset_frames[i])
        i += 1
    return new_onset_times, new_onset_frames


def generate_beatmap(file_path, melody_extract_flag=False):
    file_name = os.path.splitext(file_path)[0]
    if os.path.isfile(file_name + '_beatmap.txt'):
        print(f"The beatmap file {file_path} already exists.")
    else:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f'duration: {duration}')
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr,
                                                  hop_length=100,
                                                  backtrack=False,
                                                  units='samples')
        onset_times = librosa.samples_to_time(onset_frames)
        onset_times -= 0.6  # Time for notes to fall
        onset_times = [x for x in onset_times if x >= 0]
        notes_num = len(onset_times)
        print(f'number of notes: {notes_num}')

        avg_rate = notes_num / duration
        print(f"Average rate: {avg_rate}")

        if avg_rate > 3 and melody_extract_flag:
            traj, Z, T_coef, F_coef_hertz, F_coef_cents = libfmp.c8.compute_traj_from_audio(y, Fs=sr, constraint_region=None, gamma=0.1)
            N = 2048
            H = N // 4
            x_mel, x_acc = separate_melody_accompaniment(y, Fs=sr, N=N, H=H, traj=traj, n_harmonics=30, tol_cent=50)

            onset_frames = librosa.onset.onset_detect(y=x_mel, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames)
            onset_times -= 0.6
            onset_times = [x for x in onset_times if x >= 0]
            new_notes_num = len(onset_times)
            print(f'number of notes after melody extraction: {new_notes_num}')
            if new_notes_num > notes_num:
                onset_times = random_discard(onset_times)
        elif avg_rate > 3 and not melody_extract_flag:
            onset_times, onset_frames = random_discard(onset_times, onset_frames)
            print(f'number of notes after random discard: {len(onset_times)}')

        def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
            r = librosa.autocorrelate(segment)
            r[:int(sr / fmax)] = 0
            r[int(sr / fmin):] = 0
            loc = r.argmax()
            f0 = float(sr / loc) if loc > 0 else fmin
            return f0

        def estimate_pitch_by_onset(x, onset_samples, i, sr):
            n0 = onset_samples[i]
            n1 = onset_samples[i + 1]
            return estimate_pitch(x[n0:n1], sr=sr)

        y_pitch = np.array([
            estimate_pitch_by_onset(y, onset_frames, i, sr=sr)
            for i in range(len(onset_frames) - 1)
        ])

        min_pitch = np.min(y_pitch)
        max_pitch = np.max(y_pitch)
        sd_pitch = np.std(y_pitch)
        interval_pitch = (max_pitch - min_pitch) / 4.0
        range_pitch = [min_pitch, min_pitch + interval_pitch, min_pitch + interval_pitch * 2, min_pitch + interval_pitch * 3]

        def diff(a, b):
            return abs(a - b)

        def between(num, this):
            for temp in range(len(this) - 1):
                if num >= this[temp] and num <= this[temp + 1]:
                    return temp
            return len(this) - 1

        with open(file_name + '_beatmap.txt', 'w') as f:
            cur_col_index = between(y_pitch[0], range_pitch)
            ref_pitch = y_pitch[0]
            for i in range(len(onset_times) - 1):
                pitch_diff = diff(y_pitch[i], ref_pitch)
                step = int(pitch_diff / (sd_pitch / 3))
                cur_col_index = int(cur_col_index + step) if y_pitch[i] >= ref_pitch else int(cur_col_index - step)
                ref_pitch = y_pitch[i]
                if cur_col_index < 0 or cur_col_index > 3:
                    cur_col_index = between(y_pitch[0], range_pitch)
                f.write(f'{onset_times[i]:.4f}, {str(key_list[cur_col_index])} \n')


def get_color_from_mood_detection(file_path):
    def extract_features(audio_file):
        y, sr = librosa.load(audio_file)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_var = np.var(spectral_contrast)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)

        # Build feature dictionary
        features = {
            'tempo': tempo.reshape(1,)[0] if isinstance(tempo, np.ndarray) else tempo,
            'beat_frames': beat_frames.mean(axis=0) if len(beat_frames) > 0 else 0,
            'chroma_stft_mean': chroma_stft_mean,
            'chroma_stft_var': chroma_stft_var,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'zero_crossing_rate_var': zero_crossing_rate_var,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_var': spectral_centroid_var,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_var': spectral_bandwidth_var,
            'spectral_contrast_mean': spectral_contrast_mean,
            'spectral_contrast_var': spectral_contrast_var,
        }

        # Fixed original bug: mfccX_var was pointing to mfccs_mean
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = mfccs_mean[i]
            features[f'mfcc{i+1}_var'] = mfccs_var[i]

        # Ensure consistent feature order matching the training notebook
        feature_order = [
            'tempo', 'beat_frames', 'chroma_stft_mean', 'chroma_stft_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'spectral_contrast_mean', 'spectral_contrast_var'
        ]
        for i in range(20):
            feature_order.append(f'mfcc{i+1}_mean')
            feature_order.append(f'mfcc{i+1}_var')
            
        return [features[k] for k in feature_order]

    features_unseen_song = extract_features(file_path)

    # Load ML models
    loaded_model_arousal = joblib.load(get_asset_path('LR_arousal.sav'))
    arousal_prediction = loaded_model_arousal.predict([features_unseen_song])

    loaded_model_valence = joblib.load(get_asset_path('RF_valence.sav'))
    valence_prediction = loaded_model_valence.predict([features_unseen_song])

    arousal_val = arousal_prediction[0] if hasattr(arousal_prediction, '__iter__') else arousal_prediction
    valence_val = valence_prediction[0] if hasattr(valence_prediction, '__iter__') else valence_prediction

    if arousal_val > 0 and valence_val > 0:
        mood_color = (253, 242, 204)  # Yellow
    elif arousal_val > 0 and valence_val < 0:
        mood_color = (194, 170, 224)  # Purple
    elif arousal_val < 0 and valence_val > 0:
        mood_color = (106, 247, 209)  # Green
    else:
        mood_color = (117, 193, 240)  # Blue

    return mood_color, arousal_val, valence_val


# --- Audio Loading Initialization ---

# Choose file path using Tkinter dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select song file",
    filetypes=(("wav and mp3 files", "*.wav;*.mp3"), ("wav audio files", "*.wav"), ("mp3 audio files", "*.mp3"))
)

if not file_path:
    print("No audio file selected. Exiting.")
    sys.exit()

print(f"Loading song: {file_path}")
file_name = os.path.splitext(file_path)[0]

# --- Audio compatibility fix using librosa ---
y, sr = librosa.load(file_path, sr=None, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
frequency = sr
framerate = sr

# Generate visualizer stereo wave data
if y.ndim == 1:
    y_stereo = np.vstack((y, y))
    nchannels = 2
else:
    y_stereo = y
    nchannels = y.shape[0]

wave_data = (y_stereo * 32767).astype(np.short)
nframes = wave_data.shape[1]

# Set up visualizer details
plot_width = WIDTH
plot_height = HEIGHT - 300
buffer_size = int(1 / FPS * sr)
bar_spacing = 2
playback_speed = 1.0

# Mood and ML values
mood_color, arousal_val, valence_val = get_color_from_mood_detection(file_path)
print(f"Arousal: {arousal_val:.4f}, Valence: {valence_val:.4f}")

# Pre-generate beatmap
generate_beatmap(file_path)

# Clock & Timing
fpsclock = pg.time.Clock()
offset_time = 0


# --- Pygame Main Game Object ---

class Game:
    def __init__(self):
        self.state = "intro"
        pg.init()

        # Fonts
        self.basic_font = pg.font.SysFont("Arial", 15)
        self.title_font = pg.font.SysFont("Arial", 15)
        self.combo_font = pg.font.SysFont("Arial", 30)

        # Game stats
        self.score_val = 0
        self.combo_val = 0
        self.speed_val = 5

        # Screen & Background
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.start_background = pg.image.load(get_asset_path("start_background.png"))
        self.main_background = pg.image.load(get_asset_path("main_background.png"))

        # Visual highlights
        # Removed old image-based lane highlights as they are now drawn as rectangles

        # Grades
        self.perfect = pg.image.load(get_asset_path("Perfect.png"))
        self.good = pg.image.load(get_asset_path("Good.png"))
        self.poor = pg.image.load(get_asset_path("Poor.png"))
        self.miss = pg.image.load(get_asset_path("Miss.png"))
        self.grade = self.perfect

        # Note assets
        # Removed old image-based notes as they are now drawn as rectangles

        # Other elements
        self.scoreText = pg.image.load(get_asset_path("Score.png"))
        self.comboText = pg.image.load(get_asset_path("Combo.png"))

        self.value = False
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()

        # AI & CV configurations
        self.tracker = CameraTracker(self.screen.get_width(), self.screen.get_height())
        self.tracker.start()

        self.uninformed_agent = UninformedSearchAgent()
        self.informed_agent = InformedSearchAgent(arousal=arousal_val, valence=valence_val)
        self.fuzzy_controller = FuzzyDifficultyController()

        self.TILE_HEIGHT = 120 # Tinggi kotak hitam (Piano Tile)
        self.autoplay_enabled = False
        self.bot_lane = 1

        # Note tracking lists
        self.active_notes = []
        self.bonus_notes = []
        self.next_note_idx = 0

        # Lanes highlighted status per frame
        self.lane_highlighted = [False, False, False, False]

    def read_beatmap(self):
        with open(file_name + "_beatmap.txt", 'r') as f:
            onset_times = []
            onset_num = []
            for line in f:
                parts = line.strip().split(',')
                onset_times.append(float(parts[0]))
                onset_num.append(int(parts[1]))
        return onset_times, onset_num

    def load_beatmap(self):
        self.onset_times, self.onset_nums = self.read_beatmap()
        self.num_notes = len(self.onset_times)
        
        # Load beats from librosa for custom A* bonus note spawning
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self.beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        self.beat_times = [bt - 0.6 for bt in self.beat_times if bt - 0.6 >= 0]
        self.next_bonus_beat_idx = 4
        self.last_bonus_lane = 1

    def get_current_time(self):
        return max(0, pygame.mixer.music.get_pos() / 1000)

    def waveform_drawing(self):
        current_time = self.get_current_time()
        start_sample = int(current_time * sr * playback_speed)
        audio_data = y[start_sample:start_sample + buffer_size]

        if len(audio_data) < 2:
            return

        pygame.draw.line(self.screen, mood_color, (0, plot_height // 2), (plot_width, plot_height // 2))
        for i in range(len(audio_data) - 1):
            x1 = int(i * plot_width / len(audio_data))
            x2 = int((i + 1) * plot_width / len(audio_data))
            y1 = int((audio_data[i] + 1) * plot_height / 2)
            y2 = int((audio_data[i + 1] + 1) * plot_height / 2)
            pygame.draw.line(self.screen, mood_color, (x1, y1), (x2, y2), 2)

    def _draw_game_info(self, num):
        # Calculate reaction map using discrete cosine transform (DCT)
        num = int(num)
        start_idx = nframes - num
        if start_idx < 0:
            start_idx = 0
        height_map = abs(dct(wave_data[0][start_idx:start_idx + 4]))
        height_map = [min(HEIGHT2, int(i ** (1 / 2.5) * HEIGHT2 / 100)) for i in height_map]
        
        # Draw dynamic Reactive Equalizer columns in the background
        for idx, h in enumerate(height_map):
            bar_h = int(min(180, h))
            bar_surface = pg.Surface((88, bar_h), pg.SRCALPHA)
            bar_surface.fill((*mood_color, 70))
            self.screen.blit(bar_surface, (LANE_X[idx], HEIGHT - bar_h))

    def draw_game_info(self):
        self.num -= framerate / 60
        if self.num > 0:
            self._draw_game_info(self.num)

        # Draw static texts and panels
        self.screen.blit(self.scoreText, (207, 9))
        self.screen.blit(self.comboText, (15, 9))
        self.screen.blit(self.grade, (202, 130))
        self.screen.blit(self.score, (235, 66))
        self.screen.blit(self.combo, (21, 41))

        # Progress bar
        curr_progress = self.get_current_time() / duration
        pygame.draw.rect(self.screen, mood_color, [0, 0, 448 * curr_progress, 10])

        # Draw overlays or status text
        if self.autoplay_enabled:
            autoplay_txt = self.basic_font.render("AUTOPLAY: ACTIVE (BFS)", True, (255, 100, 100))
            self.screen.blit(autoplay_txt, (15, 80))
        else:
            autoplay_txt = self.basic_font.render("Press 'B' for Auto-Play", True, (200, 200, 200))
            self.screen.blit(autoplay_txt, (15, 80))

        if self.tracker.overlay_enabled:
            cam_txt = self.basic_font.render("CAMERA OVERLAY: ON ('C')", True, (100, 255, 100))
            self.screen.blit(cam_txt, (350, 80))
        else:
            cam_txt = self.basic_font.render("Press 'C' for Camera Overlay", True, (200, 200, 200))
            self.screen.blit(cam_txt, (320, 80))

        pygame.display.flip()

    def update_notes(self):
        curr_time = self.get_current_time()
        # Dynamically update speed using Fuzzy Logic
        self.speed_val = self.fuzzy_controller.compute_speed(self.combo_val, arousal_val)

        # 1. Spawn Normal Notes based on beatmap onset times
        while self.next_note_idx < len(self.onset_times) and curr_time >= self.onset_times[self.next_note_idx]:
            x_pos = self.onset_nums[self.next_note_idx]
            lane_idx = X_TO_LANE.get(x_pos, 0)
            self.active_notes.append({
                'lane_idx': lane_idx,
                'y': 233.0,
                'type': 'normal'
            })
            self.next_note_idx += 1

        # 2. Spawn Bonus Notes based on beats and A* path planning
        if len(self.beat_times) > 0 and self.next_bonus_beat_idx < len(self.beat_times):
            if curr_time >= self.beat_times[self.next_bonus_beat_idx]:
                # Identify where to guide the pathing
                curr_player_lane = self.bot_lane if self.autoplay_enabled else 1
                if not self.autoplay_enabled and self.tracker.finger_x is not None:
                    # Map finger coordinates to nearest lane using proper lane boundaries
                    fx = self.tracker.finger_x
                    curr_player_lane = int(fx / 128)
                    if curr_player_lane < 0:
                        curr_player_lane = 0
                    elif curr_player_lane > 3:
                        curr_player_lane = 3

                # Calculate paths using Informed A* search
                path = self.informed_agent.find_bonus_path(
                    start_lane=self.last_bonus_lane,
                    player_lane=curr_player_lane,
                    path_length=5
                )

                # Spawn 5 notes on consecutive beats
                for step, lane in enumerate(path):
                    b_idx = self.next_bonus_beat_idx + step
                    if b_idx < len(self.beat_times):
                        self.bonus_notes.append({
                            'lane_idx': lane,
                            'spawn_time': self.beat_times[b_idx],
                            'y': 233.0,
                            'type': 'bonus',
                            'hit': False
                        })
                        self.last_bonus_lane = lane

                self.next_bonus_beat_idx += 8  # Schedule next sequence in 8 beats

        # 3. Update normal notes (falling)
        notes_to_remove = []
        for note in self.active_notes:
            note['y'] += self.speed_val
            # If notes reach bottom limit without hitting
            if note['y'] >= 494:
                self.grade = self.miss
                self.combo_val = 0
                notes_to_remove.append(note)

        for note in notes_to_remove:
            self.active_notes.remove(note)

        # 4. Update bonus notes (falling)
        for bn in self.bonus_notes:
            if curr_time >= bn['spawn_time'] and not bn['hit']:
                bn['y'] += self.speed_val

    def draw_game_elements(self):
        curr_time = self.get_current_time()
        
        # 1. HAPUS ATAU KOMENTARI BARIS INI AGAR TIDAK MENUTUPI KAMERA
        # self.screen.fill((255, 255, 255)) 
        
        for i in range(1, 4):
            pg.draw.line(self.screen, (200, 200, 200), (i * 128, 0), (i * 128, HEIGHT), 2)

        # 2. Gambar tuts piano di bagian bawah (Area Hit)
        for idx in range(4):
            # Jika jalur sedang di-hit oleh AI, Keyboard, atau Jari Kamera, warnai abu-abu terang
            if self.lane_highlighted[idx]:
                pg.draw.rect(self.screen, (220, 220, 220), (LANE_X[idx], 400, 128, 112))
            
            # Gambar garis batas area hit
            pg.draw.rect(self.screen, (100, 100, 100), (LANE_X[idx], 400, 128, 112), 2)

        # 3. Render nada jatuh sebagai kotak hitam panjang (Piano Tiles)
        for note in self.active_notes:
            tile_rect = (LANE_X[note['lane_idx']] + 4, int(note['y']) - self.TILE_HEIGHT, 120, self.TILE_HEIGHT)
            pg.draw.rect(self.screen, (20, 20, 20), tile_rect, border_radius=5) # Kotak hitam dengan sudut sedikit melengkung

        # 4. Render bonus notes (jika ada, jadikan ubin warna emas)
        for bn in self.bonus_notes:
            if curr_time >= bn['spawn_time'] and not bn['hit'] and bn['y'] <= 494:
                tile_rect = (LANE_X[bn['lane_idx']] + 4, int(bn['y']) - self.TILE_HEIGHT, 120, self.TILE_HEIGHT)
                pg.draw.rect(self.screen, (255, 215, 0), tile_rect, border_radius=5)

        # 5. Render Garis Kritis (Batas Bawah Hit)
        pg.draw.line(self.screen, (255, 50, 50), (0, 485), (WIDTH, 485), 4)

        # 6. Gambar kursor virtual dari Computer Vision (Jari)
        if self.tracker.finger_x is not None:
            fx = self.tracker.finger_x
            fy = self.tracker.finger_y
            # Lingkaran jari 1 (Merah Terang agar kontras)
            pg.draw.circle(self.screen, (255, 50, 50), (fx, fy), 12)
            pg.draw.circle(self.screen, (255, 255, 255), (fx, fy), 6)

        if self.tracker.finger_x2 is not None:
            fx2 = self.tracker.finger_x2
            fy2 = self.tracker.finger_y2
            # Lingkaran jari 2 (Biru Terang agar kontras)
            pg.draw.circle(self.screen, (50, 100, 255), (fx2, fy2), 12)
            pg.draw.circle(self.screen, (255, 255, 255), (fx2, fy2), 6)

    def trigger_hit(self, lane_idx):
        # Check normal notes first
        target_note = None
        min_dist = float('inf')
        for note in self.active_notes:
            if note['lane_idx'] == lane_idx:
                dist = abs(note['y'] - 485) # Jarak ke garis kritis
                if dist < min_dist:
                    min_dist = dist
                    target_note = note

        if target_note and min_dist <= 45:
            # Score calculation based on distance to critical line
            if min_dist < 10:
                self.grade = self.perfect
                self.score_val += 1000 * max(1, self.combo_val)
            elif min_dist <= 22:
                self.grade = self.good
                self.score_val += 500 * max(1, self.combo_val)
            else:
                self.grade = self.poor
                self.score_val += 300 * max(1, self.combo_val)

            self.combo_val += 1
            self.active_notes.remove(target_note)
            self.value = True
            return True

        # Check bonus notes
        curr_time = self.get_current_time()
        target_bn = None
        min_bn_dist = float('inf')
        for bn in self.bonus_notes:
            if bn['lane_idx'] == lane_idx and curr_time >= bn['spawn_time'] and not bn['hit']:
                dist = abs(bn['y'] - 485)
                if dist < min_bn_dist:
                    min_bn_dist = dist
                    target_bn = bn

        if target_bn and min_bn_dist <= 45:
            self.grade = self.perfect
            self.score_val += 2000 * max(1, self.combo_val)  # 2x Score for bonus notes!
            self.combo_val += 1
            target_bn['hit'] = True
            self.value = True
            return True

        return False

    def process_ai_autoplay(self):
        """
        Calculates pathing using BFS to control the autoplay bot.
        """
        curr_time = self.get_current_time()
        # Find active notes that are close to the hit bar
        nearby_targets = []
        for note in self.active_notes:
            if note['y'] >= 350:
                nearby_targets.append(note['lane_idx'])

        for bn in self.bonus_notes:
            if curr_time >= bn['spawn_time'] and not bn['hit'] and bn['y'] >= 350:
                nearby_targets.append(bn['lane_idx'])

        # Remove duplicates
        nearby_targets = list(set(nearby_targets))

        # BFS pathfind
        path = self.uninformed_agent.find_path(self.bot_lane, nearby_targets)
        if path:
            self.bot_lane = path[0]  # Move bot lane indicator

        # Highlight bot's current lane
        self.lane_highlighted[self.bot_lane] = True

        # Trigger automatic hitting when notes are in the strike zone
        for note in self.active_notes:
            if note['lane_idx'] == self.bot_lane and 440 <= note['y'] <= 500:
                self.trigger_hit(self.bot_lane)
                break

        for bn in self.bonus_notes:
            if bn['lane_idx'] == self.bot_lane and curr_time >= bn['spawn_time'] and not bn['hit'] and 440 <= bn['y'] <= 500:
                self.trigger_hit(self.bot_lane)
                break

    def process_cv_hits(self):
        """
        Processes index finger coordinates to trigger lane hits.
        Supports both hands for dual-finger gameplay.
        """
        # Process first finger
        if self.tracker.finger_x is not None:
            fx = self.tracker.finger_x
            fy = self.tracker.finger_y

            # Only hit inside the strike vertical zone (440 - 500)
            if 430 <= fy <= 505:
                # Map fx to lane index using proper lane boundaries
                # Lane width: 512/4 = 128 pixels per lane
                # Lane 0: 0-128, Lane 1: 128-256, Lane 2: 256-384, Lane 3: 384-512
                hit_lane = int(fx / 128)
                if 0 <= hit_lane <= 3:
                    self.lane_highlighted[hit_lane] = True
                    self.trigger_hit(hit_lane)

        # Process second finger (if detected)
        if self.tracker.finger_x2 is not None:
            fx2 = self.tracker.finger_x2
            fy2 = self.tracker.finger_y2

            # Only hit inside the strike vertical zone (440 - 500)
            if 430 <= fy2 <= 505:
                # Map fx2 to lane index using proper lane boundaries
                hit_lane2 = int(fx2 / 128)
                if 0 <= hit_lane2 <= 3:
                    self.lane_highlighted[hit_lane2] = True
                    self.trigger_hit(hit_lane2)

    def intro(self):
        self.screen.blit(self.start_background, (0, 0))
        self.events()
        pg.display.flip()

    def main_game_init(self):
        pygame.mixer.init(frequency=frequency)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        self.load_beatmap()
        self.main_game()

    def main_game(self):
        global duration, offset_time
        offset_time = pygame.time.get_ticks() / 1000.0

        while True:
            self.events()

            # 1. Clear lane highlighted status for this frame
            self.lane_highlighted = [False, False, False, False]

            # 2. Render background: camera overlay or static image
            cam_frame = None
            if self.tracker.overlay_enabled:
                with self.tracker.lock:
                    if self.tracker.latest_frame is not None:
                        cam_frame = self.tracker.latest_frame.copy()

            if cam_frame is not None:
                # JIKA OVERLAY AKTIF: Tampilkan video OpenCV dari webcambu
                cam_surface = pg.image.frombuffer(cam_frame.tobytes(), (WIDTH, HEIGHT), 'RGB')
                # Membuat kamera sedikit transparan agar ubin tetap terlihat jelas
                cam_surface.set_alpha(180) 
                self.screen.fill((255, 255, 255)) # Dasar putih dulu
                self.screen.blit(cam_surface, (0, 0)) # Baru timpa kamera transparan
            else:
                # JIKA OVERLAY MATI: Baru tampilkan background putih bersih
                self.screen.fill((255, 255, 255))
            # 3. Update note positions & spawn new notes
            self.update_notes()

            # 4. Handle input highlights and hits
            if self.autoplay_enabled:
                self.process_ai_autoplay()
            else:
                # Highlight columns from keyboard inputs
                keys = pg.key.get_pressed()
                if keys[pg.K_a]:
                    self.lane_highlighted[0] = True
                if keys[pg.K_s]:
                    self.lane_highlighted[1] = True
                if keys[pg.K_k]:
                    self.lane_highlighted[2] = True
                if keys[pg.K_l]:
                    self.lane_highlighted[3] = True

                # Process CV Hits
                self.process_cv_hits()

            # 5. Draw active notes and cursor
            self.draw_game_elements()

            # 6. Draw waveform
            self.waveform_drawing()

            # 7. Render scoreboard, combo, visualizer
            self.score = self.combo_font.render(str(self.score_val), True, (255, 255, 255))
            self.combo = self.combo_font.render(str(self.combo_val), True, (246, 193, 66))

            fpsclock.tick(60)
            self.draw_game_info()

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()

                # State handling
                if self.state == "intro" and event.key == pg.K_RETURN:
                    self.state = 'main_game'

                # Main gameplay triggers
                elif self.state == "main_game":
                    # Toggle camera overlay
                    if event.key == pg.K_c:
                        self.tracker.overlay_enabled = not self.tracker.overlay_enabled
                        print(f"Camera overlay toggled: {self.tracker.overlay_enabled}")

                    # Toggle autoplay bot
                    elif event.key == pg.K_b:
                        self.autoplay_enabled = not self.autoplay_enabled
                        print(f"Auto-play toggled: {self.autoplay_enabled}")

                    # Keyboard hit keys (manual mode)
                    if not self.autoplay_enabled:
                        if event.key == pg.K_a:
                            self.trigger_hit(0)
                        elif event.key == pg.K_s:
                            self.trigger_hit(1)
                        elif event.key == pg.K_k:
                            self.trigger_hit(2)
                        elif event.key == pg.K_l:
                            self.trigger_hit(3)

    def handle_game_state(self):
        if self.state == 'intro':
            self.intro()
        elif self.state == 'main_game':
            self.main_game_init()

    def quit(self):
        self.tracker.stop()
        pg.quit()
        sys.exit()


# --- Program Entry ---
my_game = Game()
my_game.num = nframes

while True:
    my_game.handle_game_state()
