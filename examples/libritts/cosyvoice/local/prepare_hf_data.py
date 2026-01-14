import argparse
import os
import soundfile as sf
from datasets import load_dataset, Audio

SAMPLING_RATE=24000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help='Path to directory containing .arrow and dataset_info.json'
    )
    parser.add_argument(
        '--des_dir', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--token', 
        type=str, 
        required=True
    )
    args = parser.parse_args()
    print(args.input_dir)

    os.makedirs(args.des_dir, exist_ok=True)
    wav_dir = os.path.join(args.des_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # Load the arrow dataset
    dataset = load_dataset(
        args.input_dir,
        token = args.token,
        data_files={"train": "data-0000[0-1]-of-00049.arrow"},
        streaming=True
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    # We open the 4 Kaldi-style files
    with open(f"{args.des_dir}/wav.scp", 'w') as f_wav, \
         open(f"{args.des_dir}/text", 'w') as f_txt, \
         open(f"{args.des_dir}/utt2spk", 'w') as f_u2s, \
         open(f"{args.des_dir}/spk2utt", 'w') as f_s2u:
        
        spk_map = {}

        for item in dataset['train']:
            utt_id = item['_id']
            speaker = item['speaker']
            text = item['text']
            
            # Extract audio data
            # item['audio']['array'] is a numpy array
            audio_array = item['audio']['array']
            print(audio_array.shape)
            
            # Save raw audio to a physical .wav file (required for CosyVoice stages)
            wav_path = os.path.abspath(os.path.join(wav_dir, f"{utt_id}.wav"))
            sf.write(wav_path, audio_array, SAMPLING_RATE)

            # Write Kaldi-style records
            f_wav.write(f"{utt_id} {wav_path}\n")
            f_txt.write(f"{utt_id} {text}\n")
            f_u2s.write(f"{utt_id} {speaker}\n")
            
            if speaker not in spk_map: spk_map[speaker] = []
            spk_map[speaker].append(utt_id)

        for spk, utts in spk_map.items():
            f_s2u.write(f"{spk} {' '.join(utts)}\n")

if __name__ == "__main__":
    main()