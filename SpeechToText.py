import speech_recognition as sr


def bytearray_to_text(audio_bytes):
    # Initialize a recognizer object
    r = sr.Recognizer()

    # Convert the bytearray to an AudioData object
    audio = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)

    # Use the recognizer object to recognize speech in the audio data
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
