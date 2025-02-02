import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
#from utils import load_mp3_as_wav_with_librosa, spectrogram_from_image, spectrogram_from_waveform, image_from_spectrogram
import numpy as np
from diffusers.utils import load_image


# Load the model weights
pipe = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")
components = pipe.components
#print('ok pipe')
## weights are not reloaded into RAM
riffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
#print('ok riffusion pipeline')
#riffusion_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")

prompt = "melodic pop"
img_path = "/work3/s222948/data/raw/1000dataset_10seg/specs/1PhLYngBKbeDtdmDzCg3Pb_segment_4.png" # (metallica)
#img_path2 = "/zhome/15/5/181507/git/DiffAudio/src/a_soulful_saxophone_performance_in_a_smoky_bar.png" # (riffusion)
#audio_path = "/work3/s222948/data/processed/1000dataset_5/audio/1PhLYngBKbeDtdmDzCg3Pb_segment_3.wav" # (metallica)



#img = Image.open(img_path)#.convert("L")  # "L" mode converts the image to grayscale

img = load_image(img_path)
print("ok image loaded ")

generator = torch.Generator("cuda").manual_seed(1022)
print('ok generator')

specgram_cust_1 = riffusion_img2img(prompt=prompt, image=img, strength=0.47, generator=generator, guidance_scale=8.5).images[0]

### Save the new spectrogram image, thew old one and the prompt
# Name = prompt + path (after last '/')
output_p= '/work3/s222948/data/processed/Img2Img'
output_path =output_p + '/' + prompt + "_"+ img_path.split("/")[-1].split(".")[0] +".png"
specgram_cust_1.save(output_path)
# save parameters as txt
np.savez('/work3/s222948/data/processed/Img2Img/parameters.npz', strength=0.47, guidance_scale=8.5, seed=1022)
# save original image
or_img_path = img_path.split("/")[-1].split(".")[0] +".png"
print(f"Generated spectrogram saved to {output_path}")






import numpy as np
from PIL import Image

image_width = 512
sample_rate = 44100  # [Hz]
clip_duration_ms = 5000  # [ms]

bins_per_image = 512
n_mels = 512

# FFT parameters
window_duration_ms = 100  # [ms]
padded_duration_ms = 400  # [ms]
step_size_ms = 10  # [ms]

# Derived parameters
num_samples = int(image_width / float(bins_per_image) * clip_duration_ms) * sample_rate
n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
hop_length = int(step_size_ms / 1000.0 * sample_rate)
win_length = int(window_duration_ms / 1000.0 * sample_rate)

def spectrogram_from_image(
    image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
) -> np.ndarray:

    data = np.array(image).astype(np.float32)
    data = data[::-1, :, 0]
    data = 255 - data
    data = data * max_volume / 255
    data = np.power(data, 1 / power_for_image)

    return data


import torch
import torchaudio
from IPython.display import Audio

def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    num_samples=num_samples,
    sample_rate=sample_rate,
    mel_scale: bool = True,
    n_mels: int = 512,
    max_mel_iters: int = 200,
    num_griffin_lim_iters: int = 32,
    device: str = "cuda",
) -> np.ndarray:

    Sxx_torch = torch.from_numpy(Sxx).to(device)

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
            max_iter=max_mel_iters,
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform

waveform_cust_modified = waveform_from_spectrogram(spectrogram_from_image(specgram_cust_1))
audio_cust_mod = Audio(waveform_cust_modified, rate=sample_rate)
audio_cust_mod

# save audio
output_path_audio =output_p + '/' + prompt + "_"+ img_path.split("/")[-1].split(".")[0] +".wav"
torchaudio.save(output_path_audio, torch.tensor(waveform_cust_modified), sample_rate)