import pyttsx3
from speak import speak
engine = pyttsx3.init()
voices = engine.getProperty('voices')

for voice in voices:
    print("Voice:")
    print(" - ID: %s" % voice.id)
    print(" - Name: %s" % voice.name)
    print(" - Languages: %s" % voice.languages)
    print(" - Gender: %s" % voice.gender)
    print(" - Age: %s" % voice.age)
    speak("Hello, I am a computer program. I am testing the voices available on this computer.")
