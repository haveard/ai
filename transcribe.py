import whisper
from pyannote.audio import Pipeline

def transcribe_with_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    print("Transcription:\n", result["text"])
    return result

def speaker_diarization_with_pyannote(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_path)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker} from {turn.start:.1f}s to {turn.end:.1f}s")
    return diarization

def main():
    audio_path = 'path_to_your_audio_file.mp3'  # Replace with your audio file path
    transcribe_with_whisper(audio_path)
    speaker_diarization_with_pyannote(audio_path)

if __name__ == "__main__":
    main()
