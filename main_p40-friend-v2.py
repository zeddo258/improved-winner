import pyaudio
import requests
import tkinter.font as tkFont
import ollama
import usb.core
import usb.util
from MeloTTS.melo.api import TTS
import pygame
from usb_4_mic_array.tuning import Tuning
import tkinter as tk
import numpy as np
import threading
import os
import json


dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
Mic_tuning = ''
if dev:
    Mic_tuning = Tuning(dev)

# API endpoint URL
URL = "http://localhost:8080/upload-audio/"
URL_OLLAMA = "http://localhost:11435/api/chat"
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
    response = requests.post(URL, files=files)
    if response.status_code == 200:
        result = response.json().get("result", "")
        print(result)
        return result
    else:
        print(f"Error: {response.status_code}")
        return ""

convo = []
def stream_response(prompt):
    convo.append({'role': 'user', 'content':prompt})
    response = ''
    stream = ollama.chat(model='llama3.1:70b', messages=convo, stream=True)
    print('\nASSISSTANT:')

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    convo.append({'role':'assisstant', 'content':response})


def start_listening(stream):
    print("Listening...")

    frames = []
    silence_frames = 0

    while silence_frames < MAX_SILENCE_FRAMES:
        data = stream.read(CHUNK)
        frames.append(data)

        if Mic_tuning.is_voice():  
            silence_frames = 0 
        else:
            silence_frames += 1  

    audio_bytes = b''.join(frames)
        
    # Convert audio to text using the server
    transcribed_text = bytes_to_text(audio_bytes)
    return transcribed_text


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Llama 3.1")

        self.root.geometry("800x600") 
        font_style = tkFont.Font(family="Microsoft YaHei", size=16)
 
        # 增加對話框的大小並設置字體
        self.text_area = tk.Text(root, height=30, width=80, font=font_style)  # 設置字體大小
        self.text_area.pack(expand=True, fill=tk.BOTH)  # 使用expand和fill使對話框填充整個窗口

        # 麥克風按鈕

        self.mic_button = tk.Button(root, text="🎤 點擊說話", font=font_style, width=20, height=2, command=self.on_mic_click)  # 設置按鈕大小
        self.mic_button.pack(pady=20)  # 使用 pady 增加上下間距
        self.counter = 0
        # 儲存對話紀錄
        system_prompt = "你所在地區是台灣,你使用繁體中文對話."
        self.convo = [{"role": "system", "content":system_prompt}]

       
    def on_mic_click(self):
        # 按下按鈕時，啟動錄音並進行處理
        threading.Thread(target=self.listen_and_process).start()

    def listen_and_process(self):
        if ( self.counter == 0 ):
            first_prompt = f"""你是<speaker>,我是<listener>. 現在<speaker>是[朋友],提供幫助和聆聽對方煩惱和給對方建議.  理解<listener>的生活大小事,未來規劃,課業問題,感情問題,旅遊經驗,美食,經驗分享,各種感想,有趣的故事,幻想。回應<listener>提出的問題,若無法解答<listener>的問題，就請<listener>再更具體說明. 請用口語化的語言，與<listener>對話,理解對話內容. 以<speaker>的角色換位同理“listener”. <speaker>的回應需要排除過度積極及過度樂觀用詞. 依[朋友]的角度,使用500字以內的回應,以支持型回應方式回覆<listener>."""
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
        transcribed_text = start_listening(stream)  # 返回轉錄文本
        stream.stop_stream()
        stream.close()

        if "None" not in transcribed_text:
            # 更新用戶提問
            self.update_conversation(f"你: {transcribed_text}\n")
            # 呼叫chatbot模型處理回應
            self.process_response(transcribed_text)
    
    def update_conversation(self, message):
        self.text_area.insert(tk.END, f"{message}")
        self.text_area.see(tk.END)  # 自動滾動到最新對話

    def process_response(self, prompt):
        self.convo.append({'role': 'user', 'content': prompt})
        self.counter += 1
        response = ''
        stream = requests.post(
            url=URL_OLLAMA,
            json= {
                "model":'llama3.1:70b', 
                "messages":self.convo, 
                "stream":True,
            }
        )
        
        self.update_conversation("機器人: ") 
        for chunk in stream.iter_lines():
            body = json.loads(chunk)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done") is False:
                message = body.get("message", "")
                content = message.get("content", "")
                response += content
            self.update_conversation(content)

        self.update_conversation("\n")
        self.text_to_speech(response)
        self.convo.append({'role': 'assistant', 'content': response})

    def text_to_speech(self, text):

        # Speed is adjustable
        speed = 0.8
        device = 'cuda:0'

        model = TTS(language='ZH', device=device)
        speaker_ids = model.hps.data.spk2id

        output_path = 'zh.wav'
        model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)

        # 使用 pygame 播放音頻
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()

        # 等待音頻播放完畢
        while pygame.mixer.music.get_busy():
            continue
        # 停止並退出 pygame.mixer

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        # 刪除音頻文件
        os.remove(output_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
