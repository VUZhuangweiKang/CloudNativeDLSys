import math
import os
from tempfile import NamedTemporaryFile
from lib.DLCJob import *
import io
import librosa
import pickle
import numpy as np
import sox
import torch
from torch.utils.data import Sampler, DistributedSampler
import torchaudio
import struct

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment

torchaudio.set_audio_backend("sox_io")

# def load_audio(path):
#     sound, sample_rate = torchaudio.load(path)
#     if sound.shape[0] == 1:
#         sound = sound.squeeze()
#     else:
#         sound = sound.mean(axis=0)  # multiple channels, average
#     return sound.numpy()

def load_audio(audio_data):
    sound, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound.numpy()


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path = None, audio_data = None):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError
    
    # def compute_spectrogram(self, y):
    #     """
    #     :param y: Audio signal as an array of float numbers
    #     :return: Spectrogram of the signal
    #     """
    #     n_fft = int(self.sample_rate * self.window_size)
    #     win_length = n_fft
    #     hop_length = int(self.sample_rate * self.window_stride)
    #     # STFT
    #     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
    #                      win_length=win_length, window=self.window)
    #     spect, phase = librosa.magphase(D)
    #     # S = log(S+1)
    #     spect = np.log1p(spect)
    #     spect = torch.FloatTensor(spect)
    #     if self.normalize:
    #         mean = spect.mean()
    #         std = spect.std()
    #         spect.add_(-mean)
    #         spect.div_(std)

    #     return spect

    def compute_spectrogram(self, y):
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        # STFT
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        spect_complex = spectrogram_transform(y)

        # Compute magnitude
        spect = torch.abs(spect_complex)

        # S = log(S+1)
        spect = torch.log1p(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        if augmentation_conf and augmentation_conf.noise_dir:
            self.noise_injector = NoiseInjection(path=augmentation_conf.noise_dir,
                                                 sample_rate=self.sample_rate,
                                                 noise_levels=augmentation_conf.noise_levels)
        else:
            self.noise_injector = None

    # def parse_audio(self, audio_path = None, audio_data = None):
    #     if self.aug_conf and self.aug_conf.speed_volume_perturb:
    #         y = load_randomly_augmented_audio(audio_path, self.sample_rate)
    #     else:
    #         if audio_data is not None:
    #             y = audio_data
    #         else:
    #             y = load_audio(audio_path)

    #     if self.noise_injector:
    #         add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
    #         if add_noise:
    #             y = self.noise_injector.inject_noise(y)

    #     spect = self.compute_spectrogram(y)
        
    #     if self.aug_conf and self.aug_conf.spec_augment:
    #         spect = spec_augment(spect)

    #     return spect
    
    def parse_audio(self, audio_path = None, audio_data = None):
        if audio_data is not None:
            y = audio_data
        else:
            y = load_audio(audio_path)

        spect = self.compute_spectrogram(y)
        if self.aug_conf and self.aug_conf.spec_augment:
            spect = spec_augment(spect)
    
        return spect
    
    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(DLCJobDataset, SpectrogramParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 labels: list,
                 dtype: str = 'train',
                 normalize: bool = False,
                 aug_cfg: AugmentationConfig = None):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...
        You can also pass the directory of dataset.
        :param audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        :param dtype: dataset type (train/validation/test)
        :param labels: List containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        DLCJobDataset.__init__(self, dtype)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        SpectrogramParser.__init__(self, audio_conf, normalize, aug_cfg)

    def parse_transcript(self, txt_file):
        with open(f'/app/data/train/txt/{txt_file}', 'r') as f:
            transcript = f.read().strip('\n')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript
    
    # def _process_item(self, item_cloud_path: str, contents: Any):
    #     wav_file = item_cloud_path.split("/")[-1]
    #     txt_file = wav_file.replace("wav", "txt")
    #     wav = load_audio(contents)
    #     transcript = self.parse_transcript(txt_file)
    #     spect = self.parse_audio(audio_data=wav)
    #     return spect, transcript
    
    def _process_item(self, item_cloud_path: str, contents: Any):
        wav_file = item_cloud_path.split("/")[-1]
        txt_file = wav_file.replace("wav", "txt")
        wav, sample_rate = torchaudio.load(io.BytesIO(contents))
        transcript = self.parse_transcript(txt_file)
        spect = self.parse_audio(audio_data=wav)
        return spect, transcript


# def _collate_fn(batch):
#     def func(p):
#         return p[0].size(1)

#     batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
#     longest_sample = max(batch, key=func)[0]
#     freq_size = longest_sample.size(0)
#     minibatch_size = len(batch)
#     max_seqlength = longest_sample.size(1)
#     print(minibatch_size, freq_size, max_seqlength)

#     inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
#     input_percentages = torch.FloatTensor(minibatch_size)
#     target_sizes = torch.IntTensor(minibatch_size)
#     targets = []
#     for x in range(minibatch_size):
#         sample = batch[x]
#         tensor = sample[0]
#         target = sample[1]
#         seq_length = tensor.size(1)
#         inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)

#         input_percentages[x] = seq_length / float(max_seqlength)
#         target_sizes[x] = len(target)
#         targets.extend(target)
#     targets = torch.tensor(targets, dtype=torch.long)
#     return inputs, targets, input_percentages, target_sizes


def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(2), reverse=True)
    max_seqlength = batch[0][0].size(2)
    freq_size = batch[0][0].size(1)
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(2)
        inputs[x, :, :, :seq_length].copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_percentages, target_sizes



class AudioDataLoader(DLCJobDataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class DSRandomSampler(Sampler):
    """
    Implementation of a Random Sampler for sampling the dataset.
    Added to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, batch_size=1):
        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.start_index = 0
        self.epoch = 0
        self.batch_size = batch_size
        ids = list(range(len(self.dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def set_epoch(self, epoch):
        self.epoch = epoch


class DSElasticDistributedSampler(DistributedSampler):
    """
    Overrides the ElasticDistributedSampler to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        self.start_index = 0
        self.batch_size = batch_size
        ids = list(range(len(dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return self.num_samples


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
