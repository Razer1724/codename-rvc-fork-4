import os
import sys
import traceback

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("\n[X] MISSING LIBRARIES!")
    print("Run this in cmd: pip install numpy soundfile")
    input("Press Enter to exit...")
    sys.exit(1)

def process_audio(inputs, output_name="concatenated.wav", target_sr=48000, silence_ms=50):
    files = []
    folder_path = ""

    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        folder_path = inputs[0]
        print(f"📂 Mode: Scanning Folder ({folder_path})")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        files.sort()
    else:
        print(f"📂 Mode: Selected Files ({len(inputs)} items)")
        files = [f for f in inputs if os.path.isfile(f) and f.lower().endswith('.wav')]
        files.sort()
        
        if files:
            folder_path = os.path.dirname(files[0])

    if not files:
        print("❌ No WAV files found to process.")
        return

    print(f"🚀 Stitching {len(files)} files...")

    silence_samples = int(target_sr * (silence_ms / 1000))
    silence_buffer = np.zeros(silence_samples, dtype=np.float32)
    combined_data = []

    for i, file_path in enumerate(files):
        try:
            print(f"  Processing: {os.path.basename(file_path)}")
            data, sr = sf.read(file_path)

            # Force Mono
            if len(data.shape) > 1:
                data = data[:, 0]

            combined_data.append(data.astype(np.float32))
            if i < len(files) - 1:
                combined_data.append(silence_buffer)
                
        except Exception as e:
            print(f"  ⚠️ Skipped {os.path.basename(file_path)}: {e}")

    if combined_data:
        try:
            final_output = np.concatenate(combined_data)
            output_path = os.path.join(folder_path, output_name)
            
            sf.write(output_path, final_output, target_sr, subtype='FLOAT')
            print(f"\n✨ SUCCESS! Saved to:\n{output_path}")
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: Drag files onto the .bat file.")
        else:
            process_audio(sys.argv[1:])
    except Exception:
        print("\n❌ PYTHON SCRIPT ERROR:")
        traceback.print_exc()