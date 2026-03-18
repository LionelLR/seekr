"""Whisper-based audio and video transcription.

The :class:`Transcriber` wraps OpenAI's Whisper model with lazy loading
so that the (potentially large) model weights are only downloaded and
loaded into memory on the first call to :meth:`~Transcriber.transcribe`.
"""

from typing import Any, Dict, List, Optional

import whisper


class Transcriber:
    """Transcribes audio or video files using OpenAI Whisper.

    Whisper internally uses ffmpeg for decoding, so it accepts a wide
    variety of audio (MP3, WAV, FLAC, …) and video (MP4, MKV, …) formats.
    The model is loaded lazily on the first :meth:`transcribe` call and
    cached for the lifetime of the instance.

    Whisper performs automatic language detection, making it suitable for
    multilingual content without any extra configuration.

    Args:
        model_name: Whisper model size.  Larger models produce better
            transcriptions at the cost of speed and memory.  Available
            sizes: ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
            ``"large"``.

    Example:
        >>> tr = Transcriber("base")
        >>> segments = tr.transcribe("/path/to/audio.mp3")
        >>> segments[0]
        {'text': 'Hello world', 'start': 0.0, 'end': 2.4}
    """

    def __init__(self, model_name: str = "base", initial_prompt: str = "") -> None:
        """Initialise the transcriber without loading the model yet.

        Args:
            model_name: Whisper model variant to use.
            initial_prompt: Optional text prepended as fake prior context.
                Use this to list proper names, brands, or domain terms that
                Whisper should recognise (e.g. ``"Alice, Anthropic, GPT-4"``).
                Updated at runtime via :attr:`initial_prompt`.
        """
        self.model_name = model_name
        self.initial_prompt: str = initial_prompt
        self._model: Optional[whisper.Whisper] = None

    @property
    def model(self) -> whisper.Whisper:
        """Lazy-loaded Whisper model instance.

        Downloads and loads the model weights on first access, then
        caches the result.  Subsequent accesses return the cached instance
        without any I/O.

        Returns:
            A loaded :class:`whisper.Whisper` model ready for transcription.
        """
        if self._model is None:
            print(f"[Seekr] Loading Whisper model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Transcribe an audio or video file into timestamped segments.

        Delegates to ``whisper.Whisper.transcribe`` with ``verbose=False``
        to suppress per-segment console output.  Empty or whitespace-only
        segments are filtered out.

        Args:
            audio_path: Filesystem path to the audio or video file.
                ffmpeg must be installed and on ``PATH`` for non-WAV formats.

        Returns:
            List of segment dicts, each containing:

            - ``text`` (str): Transcribed text with leading/trailing
              whitespace stripped.
            - ``start`` (float): Segment start time in seconds.
            - ``end`` (float): Segment end time in seconds.

            Returns an empty list when no speech is detected.

        Example:
            >>> tr = Transcriber("tiny")
            >>> segs = tr.transcribe("lecture.mp3")
            >>> segs[0]["start"]
            0.0
        """
        kwargs = {"verbose": False}
        if language:
            kwargs["language"] = language
        if self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
        result = self.model.transcribe(audio_path, **kwargs)
        return [
            {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
            for seg in result.get("segments", [])
            if seg["text"].strip()
        ]
