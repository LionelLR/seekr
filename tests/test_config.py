"""Tests for app.core.config — configuration loading."""

import os

import pytest

from app.core.config import Config, load_config


class TestConfigDefaults:
    """Verify that Config has sensible default values out of the box."""

    def test_embedding_provider_default(self):
        cfg = Config()
        assert cfg.embedding_provider == "local"

    def test_whisper_model_default(self):
        cfg = Config()
        assert cfg.whisper_model == "base"

    def test_chunk_duration_default(self):
        cfg = Config()
        assert cfg.chunk_duration == 120.0

    def test_top_k_default(self):
        cfg = Config()
        assert cfg.top_k == 3

    def test_api_keys_empty_by_default(self):
        cfg = Config()
        assert cfg.openai_api_key == ""
        assert cfg.gemini_api_key == ""

    def test_cookies_from_browser_empty_by_default(self):
        cfg = Config()
        assert cfg.cookies_from_browser == ""

    def test_video_sources_empty_list(self):
        cfg = Config()
        assert cfg.video_sources == []


class TestLoadConfig:
    """Tests for the load_config() function."""

    def test_missing_file_returns_defaults(self):
        """A nonexistent path should not raise — defaults are returned."""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg.embedding_provider == "local"
        assert cfg.top_k == 3

    def test_reads_embedding_provider_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("embedding_provider: gemini\n")
        cfg = load_config(str(config_file))
        assert cfg.embedding_provider == "gemini"

    def test_reads_top_k_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("top_k: 7\n")
        cfg = load_config(str(config_file))
        assert cfg.top_k == 7

    def test_reads_chunk_duration_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("chunk_duration: 20.0\n")
        cfg = load_config(str(config_file))
        assert cfg.chunk_duration == 20.0

    def test_reads_cookies_from_browser(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cookies_from_browser: firefox\n")
        cfg = load_config(str(config_file))
        assert cfg.cookies_from_browser == "firefox"

    def test_unknown_keys_are_silently_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("completely_unknown_key: banana\ntop_k: 5\n")
        cfg = load_config(str(config_file))
        assert cfg.top_k == 5

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cfg = load_config(str(config_file))
        assert cfg.embedding_provider == "local"

    def test_env_openai_key_overrides_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("openai_api_key: from-file\n")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cfg = load_config(str(config_file))
        assert cfg.openai_api_key == "sk-from-env"

    def test_env_gemini_key_overrides_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("gemini_api_key: from-file\n")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
        cfg = load_config(str(config_file))
        assert cfg.gemini_api_key == "AIza-from-env"

    def test_data_dir_is_created(self, tmp_path):
        data_dir = str(tmp_path / "new_subdir" / "data")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"data_dir: {data_dir}\n")
        cfg = load_config(str(config_file))
        import pathlib
        assert pathlib.Path(cfg.data_dir).exists()

    def test_video_sources_list_parsed(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("video_sources:\n  - https://youtube.com/watch?v=abc\n")
        cfg = load_config(str(config_file))
        assert cfg.video_sources == ["https://youtube.com/watch?v=abc"]
