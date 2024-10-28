import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC, Wav2Vec2FeatureExtractor, WavLMForXVector
from evaluate import load
from tqdm import tqdm
from einops import rearrange, reduce, repeat
import sox
import math


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def trim_silence(wav, sample_rate):
    np_arr = wav.numpy().squeeze()
    # create a transformer
    tfm = sox.Transformer()
    tfm.silence(location=-1, silence_threshold=.1)
    # transform an in-memory array and return an array
    y_out = tfm.build_array(input_array=np_arr, sample_rate_in=sample_rate)
    duration = len(y_out)/sample_rate
    if duration < .5:
        return wav
    
    return torch.tensor(y_out).unsqueeze(0)


@torch.inference_mode()
def compute_wer(wavpath_list, text_list, wav_list=None, model_id='facebook/hubert-large-ls960-ft', truncate=False):
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = HubertForCTC.from_pretrained(model_id).to('cuda')
    model.eval()

    if wav_list is None:
        wav_list = [torchaudio.load(wavpath) for wavpath in wavpath_list]
    waveform_list = []
    sample_rate_list = []
    for waveform, sample_rate in wav_list:
        waveform_list.append(waveform)
        sample_rate_list.append(sample_rate)


    asr_text = []
    for i in tqdm(range(len(wav_list))):
        waveform = rearrange(waveform_list[i].squeeze(), 'l -> () l')
        waveform = torchaudio.functional.resample(waveform, sample_rate_list[0], processor.feature_extractor.sampling_rate)
        if truncate:
            waveform = trim_silence(waveform, processor.feature_extractor.sampling_rate)
        input_values = processor(waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt").input_values.squeeze(0).to('cuda') 
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = [transcript.lower().strip() for transcript in processor.batch_decode(predicted_ids)]

        asr_text.extend(transcription)

    if text_list is None:
        return asr_text
    print(f'asr_text: {asr_text[:3]}')
    print(f'text_list: {text_list[:3]}')
    wer = load("wer")
    wer_score = wer.compute(predictions=asr_text, references=text_list)
    return wer_score
