import pyttsx3

def list_spanish_voices():
    engine = pyttsx3.init()  # Initialize the TTS engine
    voices = engine.getProperty('voices')  # Get available voices
    spanish_voices = [(voice.id, voice.name) for voice in voices if 'spanish' in voice.name.lower() or 'español' in voice.name.lower()]

    # List all found Spanish voices
    print("Available Spanish voices:")
    for index, (voice_id, voice_name) in enumerate(spanish_voices):
        print(f"{index}: {voice_name}")

    return spanish_voices

def select_and_use_voice(spanish_voices, selected_index):
    engine = pyttsx3.init()
    voice_id, voice_name = spanish_voices[selected_index]  # Select the voice by index
    print(f"Using voice: {voice_name}")
    engine.setProperty('voice', voice_id)
    test_phrase = "Hola, esto es una prueba de texto a voz en español."
    engine.say(test_phrase)
    engine.runAndWait()

if __name__ == "__main__":
    spanish_voices = list_spanish_voices()
    selected_index = 1  # Change this to the index of the voice you want to use
    if spanish_voices and 0 <= selected_index < len(spanish_voices):
        select_and_use_voice(spanish_voices, selected_index)
    else:
        print("Invalid voice selection.")