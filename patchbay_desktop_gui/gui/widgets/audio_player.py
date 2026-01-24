"""Audio playback abstraction for the DearPyGui frontend.

This module provides a small helper that supports typical transport functions
needed by the GUI:

- Play / Stop / Rewind
- Seek to an absolute position (seconds)
- Step forward/backward by a given amount (seconds)

Implementation details
----------------------
We avoid QtMultimedia to eliminate Qt DLL/plugin issues on Windows.
Playback is implemented using:

- ``soundfile`` for reading audio files into numpy arrays
- ``sounddevice`` for streaming audio to the default output device

The player uses a callback-based :class:`sounddevice.OutputStream`. The GUI
thread controls transport operations via a lock-protected shared state.

Important: This is a best-effort playback layer for desktop tooling.
It is not meant to be a full DAW player. Latency and device selection are
delegated to PortAudio/sounddevice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PlaybackState:
    """Snapshot of current playback state."""

    playing: bool
    position_samples: int
    total_samples: int
    sample_rate: int
    channels: int


class AudioPlayer:
    """A simple stream-based audio player with seeking.

    Usage pattern:
    -------------
    - Call :meth:`load_buffer` with a float32 numpy array.
    - Use :meth:`play`, :meth:`stop`, :meth:`rewind`, :meth:`seek_seconds`.
    - Poll :meth:`get_state` from the GUI (e.g., every frame) to update the
      playhead line.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._buffer: Optional[np.ndarray] = None  # shape (N, C)
        self._sr: int = 48000
        self._pos: int = 0
        self._playing: bool = False
        self._stream = None
        self._last_stream_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Buffer / stream management
    # ------------------------------------------------------------------

    def load_buffer(self, audio: np.ndarray, sample_rate: int) -> None:
        """Load an audio buffer.

        Parameters
        ----------
        audio:
            Float32 array of shape (N,) for mono or (N,C) for multi-channel.
            Values should be in [-1, 1] for best results.
        sample_rate:
            Sample rate in Hz.
        """
        if audio is None:
            raise ValueError("audio buffer is None")
        if audio.ndim == 1:
            audio2 = audio[:, None]
        elif audio.ndim == 2:
            audio2 = audio
        else:
            raise ValueError(f"audio must be 1D or 2D, got ndim={audio.ndim}")

        if audio2.dtype != np.float32:
            audio2 = audio2.astype(np.float32, copy=False)

        # IMPORTANT (Windows / PortAudio / multiple sample rates)
        # -----------------------------------------------
        # An OutputStream is created with a *fixed* samplerate and channel
        # count. If we keep an existing stream open and then load a buffer
        # with a *different* sample rate, playback will sound too fast/slow
        # because the stream continues to run at the old samplerate.
        #
        # This can happen frequently in this application because:
        # - Input files may be 44.1 kHz or 48 kHz.
        # - Output files are typically written at the model sample rate
        #   (often 16 kHz / 32 kHz depending on the SAM-Audio checkpoint).
        #
        # Therefore we ALWAYS close (tear down) any existing stream whenever
        # a new buffer is loaded. The stream is re-created lazily on the next
        # call to play().
        self.close()

        with self._lock:
            self._buffer = np.ascontiguousarray(audio2)
            self._sr = int(sample_rate)
            self._pos = 0
            self._playing = False
            self._last_stream_error = None

    def close(self) -> None:
        """Close the output stream if open."""
        # Do NOT keep the lock while stopping/closing the PortAudio stream.
        # The audio callback may attempt to acquire the same lock.
        # Holding the lock here can cause deadlocks on some systems.
        stream = None
        with self._lock:
            self._playing = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start (or resume) playback.

        Usability tweak:
        - If the playhead is at (or beyond) the end of the buffer, we restart
          from the beginning. This matches typical media player behaviour and
          avoids the impression that playback is "broken" when the user presses
          Play at end-of-file.
        """
        with self._lock:
            if self._buffer is None:
                return
            # Auto-rewind at end-of-file.
            try:
                n = int(self._buffer.shape[0])
                if self._pos >= n:
                    self._pos = 0
            except Exception:
                pass
            self._ensure_stream_locked()
            # If stream creation failed, _ensure_stream_locked() will set
            # _last_stream_error and keep _playing=False.
            if self._stream is None and self._last_stream_error:
                self._playing = False
                return
            self._playing = True

    def stop(self) -> None:
        """Stop playback (position is kept)."""
        with self._lock:
            self._playing = False

    def rewind(self) -> None:
        """Stop playback and jump to start."""
        with self._lock:
            self._playing = False
            self._pos = 0

    def seek_seconds(self, t: float) -> None:
        """Seek to absolute time in seconds."""
        with self._lock:
            if self._buffer is None:
                return
            t = float(t)
            n = self._buffer.shape[0]
            pos = int(round(t * self._sr))
            self._pos = max(0, min(pos, n))

    def step_seconds(self, delta: float) -> None:
        """Seek relatively by delta seconds."""
        with self._lock:
            if self._buffer is None:
                return
            n = self._buffer.shape[0]
            self._pos = max(0, min(self._pos + int(round(float(delta) * self._sr)), n))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> PlaybackState:
        with self._lock:
            if self._buffer is None:
                return PlaybackState(False, 0, 0, self._sr, 1)
            return PlaybackState(
                playing=bool(self._playing),
                position_samples=int(self._pos),
                total_samples=int(self._buffer.shape[0]),
                sample_rate=int(self._sr),
                channels=int(self._buffer.shape[1]),
            )

    def get_last_error(self) -> Optional[str]:
        return self._last_stream_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_stream_locked(self) -> None:
        """Create and start the stream if needed.

        Must be called with self._lock held.
        """
        # If a stream already exists we normally keep it.
        #
        # However, PortAudio can stop a stream (e.g. after reaching EOF when
        # the callback raises CallbackStop). In that case `self._stream` is
        # still non-None but no audio will be produced on subsequent Play
        # presses unless we re-create the stream.
        if self._stream is not None:
            try:
                active = bool(getattr(self._stream, "active", True))
            except Exception:
                active = True
            if active:
                return
            # Stream exists but is not active anymore -> tear down and recreate.
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Import sounddevice lazily so the GUI can still start if playback
        # dependencies are missing.
        try:
            import sounddevice as sd
        except Exception as e:
            self._last_stream_error = f"sounddevice import failed: {e}"
            self._playing = False
            return

        if self._buffer is None:
            return

        channels = int(self._buffer.shape[1])
        sr = int(self._sr)

        def callback(outdata, frames, time_info, status):
            # This callback is executed in PortAudio's thread.
            # Keep it short and lock-free as much as possible.
            try:
                if status:
                    # status can be non-fatal; do not spam UI.
                    pass
                with self._lock:
                    buf = self._buffer
                    if buf is None:
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    if not self._playing:
                        outdata[:] = 0
                        return

                    n = buf.shape[0]
                    pos = self._pos
                    end = min(pos + frames, n)
                    chunk = buf[pos:end]

                    # Fill output buffer.
                    outdata[: len(chunk), :channels] = chunk
                    if len(chunk) < frames:
                        outdata[len(chunk) :, :channels] = 0
                        self._pos = n
                        self._playing = False
                        raise sd.CallbackStop()
                    else:
                        self._pos = end
            except Exception:
                # In case of unexpected errors, stop playback.
                with self._lock:
                    self._playing = False
                raise

        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                callback=callback,
                finished_callback=None,
                blocksize=0,  # let PortAudio decide
            )
            stream.start()
            self._stream = stream
        except Exception as e:
            self._last_stream_error = f"failed to start OutputStream: {e}"
            self._playing = False
            self._stream = None


def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file via soundfile.

    Returns
    -------
    (audio, sr)

    audio is float32 numpy array of shape (N,C) (always 2D).
    """
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(
            "soundfile is required for GUI playback and waveform display. "
            "Install with: pip install soundfile"
        ) from e

    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile returns (N,C)
    return np.ascontiguousarray(audio), int(sr)


def mono_mix(audio_2d: np.ndarray) -> np.ndarray:
    """Downmix to mono for waveform display."""
    if audio_2d.ndim != 2:
        raise ValueError("audio_2d must be 2D")
    if audio_2d.shape[1] == 1:
        return audio_2d[:, 0]
    return np.mean(audio_2d, axis=1)


def now_seconds() -> float:
    """Small helper used by the GUI for timekeeping."""
    return time.time()
