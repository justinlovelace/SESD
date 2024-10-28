from praatio import textgrid
import os

data_path = '../data/aligned_librispeech/dev-clean/84/84-121123-0002.TextGrid'

# tg = textgrid.openTextgrid(data_path, False)
def get_word_intervals(textgrid_path):
    tg = textgrid.openTextgrid(textgrid_path, False)
    return tg.getTier("words").entries

def get_partial_transcript(textgrid_path, transcript_end_time=3):
    intervals = get_word_intervals(textgrid_path)
    word_list = []
    start_time = 0
    end_time = intervals[-1].end
    # Reverse intervals to get the last word first
    intervals = intervals[::-1]
    for interval in intervals:
        if (end_time - interval.start) > transcript_end_time:
            break
        word_list.append(interval.label)
        start_time = interval.start
    # Reverse word_list to get the words in the correct order
    word_list = word_list[::-1]
    return {'transcript': ' '.join(word_list), 
            'end_time': end_time,
            'start_time': start_time,}


if __name__ == "__main__":
    partial_transcript = get_partial_transcript(data_path)
    print(partial_transcript)