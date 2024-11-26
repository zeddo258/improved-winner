
URL = "http://localhost:8080/upload-audio/"
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1 
CHUNK = 1024
RECORD_SECONDS = 2
MAX_SILENCE_FRAMES = 30

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

def start_listening():
    stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,
        )
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
    stream.stop_stream()
    stream.close()

    return transcribed_text