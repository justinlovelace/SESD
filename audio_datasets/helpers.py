from audio_datasets.constants import ENCODEC_REDUCTION_FACTOR, ENCODEC_SAMPLING_RATE, MAX_DURATION_IN_SECONDS


def round_up_to_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        return number + (multiple - remainder)
    
def round_up_to_waveform_multiple(number, multiple=16):
    waveform_multiple = multiple*ENCODEC_REDUCTION_FACTOR
    rounded_number = round_up_to_multiple(number, waveform_multiple)
    return rounded_number

def compute_max_length(multiple=16):
    max_len = MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE

    max_len = round_up_to_waveform_multiple(max_len, multiple)
    return max_len

def is_audio_length_in_range(audio, sampling_rate=ENCODEC_SAMPLING_RATE):
    return len(audio['array']) <= (MAX_DURATION_IN_SECONDS*sampling_rate)

def is_audio_length_in_test_range(audio):
    return ((4*ENCODEC_SAMPLING_RATE) <= len(audio['array'])) and (len(audio['array']) <= (10*ENCODEC_SAMPLING_RATE))
