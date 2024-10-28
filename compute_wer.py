import os
import json
import argparse
from audio_datasets import LibriSpeech
from evaluation.evaluate_transcript import compute_wer
from utils.utils import parse_float_tuple

def get_wer(sample_dir, text_list, num_samples, guidances, prefix_seconds=0.0):
    wer_dict = {}
    for guidance in guidances:
        filepaths_list = []
        for i in range(num_samples):
            if prefix_seconds > 0:
                filepaths_list.append(os.path.join(sample_dir, f'guide{guidance:.1f}_prefix{prefix_seconds:.1f}', f'audio_{i}.wav'))
            else:
                filepaths_list.append(os.path.join(sample_dir, f'guide{guidance:.1f}', f'audio_{i}.wav'))
        
        text_list = text_list[:num_samples]
        wer = compute_wer(filepaths_list, text_list)
        wer_dict[guidance] = wer
        print(f'WER for guidance {guidance}: {wer*100:.1f}')
    return wer_dict

def main(args):
    test_ls_dataset = LibriSpeech(split='test')
    text_list = test_ls_dataset.hf_dataset['text']

    wer_dict = get_wer(args.sample_dir, text_list, args.num_samples, args.guidance, args.prefix_seconds)

    # Save wer_dict to a json file
    with open(os.path.join(args.sample_dir, 'wer.json'), 'w') as f:
        json.dump(wer_dict, f)
    # Print wer_dict
    print(wer_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--sample_dir", type=str, default='saved_models/librispeech/test/librispeech_250k/samples/step_100000')
    parser.add_argument("--num_samples", type=int, default=1237)
    parser.add_argument('--guidance', type=parse_float_tuple, help='Tuple of float values for dim_mults')
    parser.add_argument('--prefix_seconds', type=float, default=0.0)

    args = parser.parse_args()
    main(args)