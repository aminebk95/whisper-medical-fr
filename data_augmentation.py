import numpy as np

def add_noise(audio: np.ndarray, noise_factor: float = 0.003) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a NumPy array")

    audio = audio.astype(np.float32, copy=False)

    noise = np.random.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noisy_audio = audio + noise_factor * noise

    return np.clip(noisy_audio, -1.0, 1.0)


def maybe_add_noise(
    audio: np.ndarray,
    min_factor: float = 0.001,
    max_factor: float = 0.005,
    p: float = 0.35,   # 👈 Whisper-safe default
) -> np.ndarray:
    if np.random.rand() >= p:
        return audio

    noise_factor = float(np.random.uniform(min_factor, max_factor))
    return add_noise(audio, noise_factor)
