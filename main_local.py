import pyaudio
import requests
import tkinter.font as tkFont
import ollama
import usb.core
import usb.util
from MeloTTS.melo.api import TTS
from opencc import OpenCC
import pygame
import keyboard
from usb_4_mic_array.tuning import Tuning
from collections import Counter
import tkinter as tk
import numpy as np
import threading
import os
import cv2 
from threading import Lock
import io
import re 
from PIL import Image, ImageTk 


dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
Mic_tuning = ''
if dev:
    Mic_tuning = Tuning(dev)
converter = OpenCC('s2t')  # 或使用 's2tw' 轉為台灣繁體中文
# API endpoint URL
SENSEVOICE_URL = "http://localhost:8080/upload-audio/"
POSTER_URL = "http://localhost:8000/ask/"
TTS_URL = "http://localhost:5000/generate_audio"

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1 
CHUNK = 1024
RECORD_SECONDS = 2
MAX_SILENCE_FRAMES = 30


p = pyaudio.PyAudio()

  
def bytes_to_text(audio_bytes: bytes):
    files = {'file': ('microphone_audio.wav', audio_bytes, 'audio/wav')}
    response = requests.post(SENSEVOICE_URL, files=files)
    if response.status_code == 200:
        result = response.json().get("result", "")
        print(result)
        return result
    else:
        print(f"Error SenseVoice: {response.status_code}")
        return ""

convo = []
def stream_response(prompt):
    convo.append({'role': 'user', 'content':prompt})
    response = ''
    stream = ollama.chat(model='llama3.1', messages=convo, stream=True)
    print('\nASSISSTANT:')

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    convo.append({'role':'assisstant', 'content':response})



class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Llama 3.1")
        self.root.geometry("1200x600")  # Adjusted size for side-by-side layout

        self.poster_result = []  # Shared variable for the latest frame
        self.poster_result_lock = Lock()  # Lock to ensure thread-safe access to the frame

        font_style = tkFont.Font(family="Microsoft YaHei", size=16)

        # Configure grid for the root
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Chatbot Frame (Left Section)
        self.chatbot_frame = tk.Frame(self.root, bg="white")
        self.chatbot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure rows in the chatbot frame
        self.chatbot_frame.grid_rowconfigure(0, weight=1)  # Text area row
        self.chatbot_frame.grid_rowconfigure(1, weight=0)  # Button row
        self.chatbot_frame.grid_columnconfigure(0, weight=1)

        # Text Area for Chatbot
        self.text_area = tk.Text(self.chatbot_frame, font=font_style, wrap=tk.WORD)
        self.text_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Microphone Button
        self.mic_button = tk.Button(self.chatbot_frame, text="🎤 Click to Speak", font=font_style, command=self.on_mic_click)
        self.mic_button.grid(row=1, column=0, pady=5, sticky="ew")

        # Webcam Frame (Right Section)
        self.webcam_frame = tk.Frame(self.root, bg="white")
        self.webcam_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Webcam Label for Displaying Feed
        self.webcam_label = tk.Label(self.webcam_frame, bg="white")
        self.webcam_label.pack(expand=True, fill=tk.BOTH)

        self.face_emotion_label = tk.Label(
            self.webcam_frame, text="Facial emotion: ", font=font_style, bg="white", anchor="w", relief="sunken"
        )
        self.face_emotion_label.pack(fill=tk.X, padx=5, pady=5)

        self.voice_emotion_label = tk.Label(
            self.webcam_frame, text="Voice emotion: ", font=font_style, bg="white", anchor="w", relief="sunken"
        )
        self.voice_emotion_label.pack(fill=tk.X, padx=5, pady=5)

        self.voice_emotion_label = tk.Label(
            self.webcam_frame, text="Text emotion: ", font=font_style, bg="white", anchor="w", relief="sunken"
        )
        self.voice_emotion_label.pack(fill=tk.X, padx=5, pady=5)


        # 儲存對話紀錄
        system_prompt = "你所在地區是台灣,你使用繁體中文對話."
        self.convo = [{"role": "system", "content":system_prompt}]


        # Start the webcam thread
        self.location = 0
        self.running = True
        self.is_recording = False
        self.webcam_thread = threading.Thread(target=self.show_webcam_feed)
        self.webcam_thread.start()
        self.counter = 0

    def show_webcam_feed(self):
        """Continuously display webcam feed and send frames to API during recording."""
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))


                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.webcam_label.configure(image=img)
                self.webcam_label.image = img

                # If recording, process the frame
                if self.is_recording:
                    detected_emotion = self.send_frame_to_api(frame)
                    with self.poster_result_lock:
                        self.poster_result.append(detected_emotion)
            else:
                break

        cap.release()

    def send_frame_to_api(self, frame):
        # Save pil image to IO buffer
        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        buf.seek(0)

        files = {'image': ('image.jpg', buf, 'image/jpeg') }

        try: 
            r = requests.post(POSTER_URL, files=files)
            if r.status_code == 200:
                prediction = r.json().get("result")
                return prediction
        except Exception as e:
            print(f"Failed to send frame:{e}")

    def on_mic_click(self):
        """Toggle recording state and handle audio processing."""
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.mic_button.config(text="🔴 Recording... Click to Stop")
            print("Recording started")
            threading.Thread(target=self.listen_and_process).start()
        else:
            self.mic_button.config(text="🎤 Click to Speak")
            print("Recording stopped")

    def start_listening(self, stream):
        print("Listening...")

        frames = []
        while self.is_recording:
            data = stream.read(CHUNK)
            frames.append(data)
            self.location = Mic_tuning.direction

        self.update_location()
        audio_bytes = b''.join(frames)
        # Convert audio to text using the server
        transcribed_text = bytes_to_text(audio_bytes)
        print(transcribed_text)
        return transcribed_text

    def update_location(self):
        if ( 121 <= self.location <= 270 ):
            print("Turn left")
            keyboard.press('j')
            keyboard.release('j')
        elif ( 271 <= self.location <= 360 or 0 <= self.location <= 59 ):
            print("Turn right")
            keyboard.press('l')
            keyboard.release('l')

    def listen_and_process(self):
        if ( self.counter == 0 ):
            first_prompt = f"""你是“speaker”，我是“listener”。
                            現在“speaker”是(朋友)。
                            提供幫助和聆聽對方煩惱和給對方建議。
                            理解“listener”的生活大小事、未來規劃、課業問題、感情問題、旅遊經驗、美食、經驗分享、各種感想、有趣的故事、幻想。
                            回應“listener”提出的問題，簡單說明“listener”的所提出的問題。
                            若無法簡單以容易理解的對話，解答“listener”的問題，就請“listener”再更具體說明。
                            請用更口語化的語言，與“listener”對話。
                            理解對話內容以“speaker”的角色換位同理“listener”。
                            “speaker”的回應需要排除過度積極及過度樂觀用詞。
                            依(朋友)的角度，使用500字以內的回應，以支持型回應方式回覆“listener”。"""
            self.convo.append({'role': 'user', 'content': first_prompt})

        # 開始錄音，並處理轉換和回應
        stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,
        )
        
        # 呼叫錄音及處理函數
        self.update_conversation("Listening...\n")  # UI更新為正在聽
        transcribed_text = self.start_listening(stream)  # 返回轉錄文本
        emotion = ""
        text = ""
        for item in transcribed_text:
            text_content = item['text']  # Extract the string from the dictionary
            # Extract emotion
            emotion_match = re.search(r'<\|([A-Z]+)\|>', text_content)
            emotion = emotion_match.group(1) if emotion_match else "Unknown"

            # Extract text content
            text_match = re.search(r'<\|withitn\|>(.+)', text_content)
            text = text_match.group(1) if text_match else ""

        text = converter.convert(text)
        stream.stop_stream()
        stream.close()

        if "None" not in text:
            with self.poster_result_lock:
                if self.poster_result:
                    most_common_emotion = Counter(self.poster_result).most_common(1)[0][0]
                else:
                    most_common_emotion = "Neutral"
            # 更新用戶提問
            self.update_conversation(f"你: {text}\n")
            if emotion == "Unknown":
                self.voice_emotion_label.config(text=f"Voice Emotion: Neutral")
            else:
                self.voice_emotion_label.config(text=f"Voice Emotion: {emotion}")

            # 呼叫chatbot模型處理回應
            self.process_response(text, poster_emotion=most_common_emotion, sense_voice_emotion=emotion)
    
    def update_conversation(self, message):
        self.text_area.insert(tk.END, f"{message}")
        self.text_area.see(tk.END)  # 自動滾動到最新對話

    def process_response(self, prompt, poster_emotion, sense_voice_emotion):
        prompt = f"Listener臉部表情為 : {poster_emotion} 他的聲調表情為 :{sense_voice_emotion} Listener跟你說 :{prompt}"
        self.convo.append({'role': 'user', 'content': prompt})
        self.counter += 1
        response = ''
        self.face_emotion_label.config(text=f"Facial Emotion: {poster_emotion}")
        stream = ollama.chat(model='llama3.1', messages=self.convo, stream=True)
        
        self.update_conversation("機器人: ") 
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            self.update_conversation(content)
        self.update_conversation("\n")
        self.text_to_speech(response)
        self.convo.append({'role': 'assistant', 'content': response})
        self.poster_result = []

    def text_to_speech(self, text):

        # The text you want to synthesize
        data = {"text": text}

        # Send POST request
        response = requests.post(TTS_URL, json=data)

        #Check if the response is successful
        if response.status_code == 200:
            # Write the audio content to a file
            with open("output_client.wav", "wb") as audio_file:
                audio_file.write(response.content)
            print("Audio saved as 'output_client.wav'")
        else:
            print(f"Failed to generate audio. Status code: {response.status_code}, Error: {response.text}")
     

        # 使用 pygame 播放音頻
        pygame.mixer.init()
        pygame.mixer.music.load("output_client.wav")
        pygame.mixer.music.play()

        # 等待音頻播放完畢
        while pygame.mixer.music.get_busy():
            continue
        # 停止並退出 pygame.mixer

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        # 刪除音頻文件
        os.remove("./output_client.wav")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
