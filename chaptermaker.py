import argparse
from faster_whisper import WhisperModel
import google.genai as genai
from dotenv import load_dotenv
import os


load_dotenv()
api_gem_key = os.environ["GEMINI_API_KEY"]
api_open_key = os.environ["OPENROUTER_API_KEY"]
GEMINI_PROMPT = """Here's a transcript. Generate chapter titles with timestamsp at natural breakpoints for YouTube. Use keywords while thinking in YouTube SEO, GEO AND AEO, as well as human retention or interest. Don't go to far from the tone of the video transcription. They are Rehooks, should be leveraged for retention. Do not include word Rehook in chapter title. Keep each chapter title maximum 4 words or 21 characters. Choose the most relevant keywords for each chapter title. Use the following format for the output: [00:00] Chapter Title 1, [00:00] Chapter Title 2, etc. Do not include any additional text or explanations, just the chapter titles with timestamps. Choose maximum 10 chapters, only when they're relevant for the topic. Minimum 4. Here is the transcript: """



#openkey = openrouter.configure(api_gem_key) 

def main():
    parser = argparse.ArgumentParser(description='Chapter Maker')

    # Argument to specify the input video file
    parser.add_argument('-i', '--input', type=str, required=True, help='Archivo de video de entrada')

    # Argument to specify the output text file for chapters
    parser.add_argument('-o','--output',type=str, default="chapters.txt", help='Archivo de texto de salida con los capítulos')
    
    #Choosing a whisper model
    parser.add_argument('-m', '--model', choices=['base','small','medium','large'], default='base', help='Modelo de Whisper a utilizar')

    args = parser.parse_args()
    print(args)

    path = args.input
    parts = path.split("\\")
    parts1 = (parts[-1]).split(".")
    parts2 = (parts1[0])
    cachename = parts2 + ".txt"


    if  os.path.exists(cachename):
        print("Archivo de caché encontrado. Usando transcripción en caché.")
        transtr = open(cachename, "r").read()
        __file__ os.path.dirname() and os.path.join()
    else:
        transcript = []
        model = WhisperModel(args.model,device="cpu")
        segments, info = model.transcribe(args.input)
        for i in segments:
            print(i.start, i.end, i.text)
            transcript.append(i.text + " ")
        transtr = " ".join(transcript)
        print(transtr)
        with open(cachename, "w") as f:
            f.write(transtr)
    geminilogin = genai.Client(api_key=api_gem_key)
    geminiresponse = geminilogin.models.generate_content(
    model="gemini-2.5-flash",
    contents=GEMINI_PROMPT+transtr
    )
    print(geminiresponse.text)
    with open(args.output, "w") as f:
            f.write(geminiresponse.text)

if __name__ == "__main__":
    main()

    