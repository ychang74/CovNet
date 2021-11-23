sample_rate = 16000
audio_duration = 10
audio_samples = sample_rate * audio_duration
audio_duration_flusense = 1
audio_samples_flusense = sample_rate * audio_duration_flusense

window_size = 512
overlap = 256
mel_bins = 64

device = 'cuda'
num_epochs = 30
gamma = 0.4
patience = 4
step = 10
random_seed = 36851234
split_ratio = 0.2
classes_num = 2
classes_num_flusense = 9
exclude = ['burp', 'vomit', 'hiccup', 'snore', 'wheeze']
valid_labels = ['cough', 'speech', 'etc', 'silence', 'sneeze', 'gasp', 'breathe', 'sniffle', 'throat-clearing']
flusense_weights = [63.38, 2.45, 1.0, 47.78, 13.49, 27.89, 24.93, 0.91, 128.05]
