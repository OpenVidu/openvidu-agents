import ast
import importlib
import inspect
import pathlib
import re
import unittest
from unittest.mock import patch, MagicMock
import os
import logging
from io import StringIO

# Import the module to test
import sys

sys.path.append(".")  # Adjust as needed for your project structure
from azure.cognitiveservices.speech.enums import ProfanityOption
from livekit.plugins.soniox import STTOptions as SonioxSTTOptions

# Import the module containing the functions we want to test
import stt_impl

from stt_impl import (
    get_aws_stt_impl,
    get_azure_stt_impl,
    get_azure_openai_stt_impl,
    get_google_stt_impl,
    get_openai_stt_impl,
    get_groq_stt_impl,
    get_deepgram_stt_impl,
    get_assemblyai_stt_impl,
    get_fal_stt_impl,
    get_clova_stt_impl,
    get_speechmatics_stt_impl,
    get_gladia_stt_impl,
    get_sarvam_stt_impl,
    get_spitch_stt_impl,
    get_cartesia_stt_impl,
    get_soniox_stt_impl,
    get_nvidia_stt_impl,
    get_elevenlabs_stt_impl,
    get_simplismart_stt_impl,
    get_vosk_stt_impl,
    get_stt_impl,
)


class TestSTTImplementations(unittest.TestCase):
    def setUp(self):
        # Set up common test data
        self.base_config = {"live_captions": {"provider": "test_provider"}}

        # Capture logging
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        # Clean up after tests
        self.logger.removeHandler(self.handler)
        self.log_capture.close()

        # Clear environment variables that might have been set
        env_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "SPITCH_API_KEY",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    # AWS STT Tests
    @patch("livekit.plugins.aws.STT")
    def test_get_aws_stt_impl_success(self, mock_aws_stt):
        # Arrange
        config = {
            "live_captions": {
                "aws": {
                    "aws_access_key_id": "test_key_id",
                    "aws_secret_access_key": "test_secret_key",
                    "aws_default_region": "us-west-2",
                    "language": "en-US",
                    "vocabulary_name": "test_vocab",
                    "language_model_name": "test_model",
                    "enable_partial_results_stabilization": True,
                    "partial_results_stability": "high",
                    "vocab_filter_name": "test_filter",
                    "vocab_filter_method": "mask",
                }
            }
        }
        mock_aws_stt.return_value = "aws_stt_instance"

        # Act
        result = get_aws_stt_impl(config)

        # Assert
        self.assertEqual(result, "aws_stt_instance")
        mock_aws_stt.assert_called_once_with(
            region="us-west-2",
            language="en-US",
            vocabulary_name="test_vocab",
            language_model_name="test_model",
            enable_partial_results_stabilization=True,
            partial_results_stability="high",
            vocab_filter_name="test_filter",
            vocab_filter_method="mask",
        )
        self.assertEqual(os.environ["AWS_ACCESS_KEY_ID"], "test_key_id")
        self.assertEqual(os.environ["AWS_SECRET_ACCESS_KEY"], "test_secret_key")
        self.assertEqual(os.environ["AWS_DEFAULT_REGION"], "us-west-2")

    def test_get_aws_stt_impl_missing_credentials(self):
        # Arrange
        config = {"live_captions": {"aws": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_aws_stt_impl(config)

        self.assertIn("Wrong AWS credentials", str(context.exception))

    def test_get_aws_stt_impl_partial_credentials(self):
        # Arrange
        config = {
            "live_captions": {
                "aws": {
                    "aws_access_key_id": "test_key_id",
                    # Missing other required fields
                }
            }
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_aws_stt_impl(config)

        self.assertIn("Wrong AWS credentials", str(context.exception))

    # Azure STT Tests
    @patch("livekit.plugins.azure.STT")
    def test_get_azure_stt_impl_with_host(self, mock_azure_stt):
        # Arrange
        config = {
            "live_captions": {
                "azure": {
                    "speech_host": "test.host.com",
                    "language": ["en-US", "es-ES"],
                }
            }
        }
        mock_azure_stt.return_value = "azure_stt_instance"

        # Act
        result = get_azure_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_azure_stt.assert_called_once_with(
            speech_host="test.host.com", language=["en-US", "es-ES"]
        )

    @patch("livekit.plugins.azure.STT")
    def test_get_azure_stt_impl_with_key_region(self, mock_azure_stt):
        # Arrange
        config = {
            "live_captions": {
                "azure": {
                    "speech_key": "test_key",
                    "speech_region": "westus",
                    "language": ["en-US"],
                }
            }
        }
        mock_azure_stt.return_value = "azure_stt_instance"

        # Act
        result = get_azure_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_azure_stt.assert_called_once_with(
            speech_key="test_key",
            speech_region="westus",
            language=["en-US"],
        )

    @patch("livekit.plugins.azure.STT")
    def test_get_azure_stt_impl_with_token_region(self, mock_azure_stt):
        # Arrange
        config = {
            "live_captions": {
                "azure": {
                    "speech_auth_token": "test_token",
                    "speech_region": "westus",
                    "language": ["en-US"],
                    "profanity": "Masked",
                }
            }
        }
        mock_azure_stt.return_value = "azure_stt_instance"

        # Act
        result = get_azure_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_azure_stt.assert_called_once_with(
            speech_auth_token="test_token",
            speech_region="westus",
            language=["en-US"],
            profanity=ProfanityOption.Masked,
        )

    def test_get_azure_stt_impl_invalid_credentials(self):
        # Arrange
        config = {
            "live_captions": {
                "azure": {
                    # Missing required credentials
                }
            }
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_azure_stt_impl(config)

        self.assertIn(
            "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n    - speech_auth_token + speech_region",
            str(context.exception),
        )

    # Azure OpenAI STT Tests
    @patch("livekit.plugins.openai.STT.with_azure")
    def test_get_azure_openai_stt_impl_success(self, mock_azure_openai_stt):
        # Arrange
        config = {
            "live_captions": {
                "azure_openai": {
                    "azure_api_key": "test_azure_api_key",
                    "azure_ad_token": "test_azure_ad_token",
                    "azure_endpoint": "https://test.openai.azure.com/",
                    "api_version": "2024-02-01",
                    "azure_deployment": "test_deployment",
                    "organization": "test_org",
                    "project": "test_project",
                    "language": "es",
                    "detect_language": True,
                    "model": "gpt-4o-transcribe",
                    "prompt": "Transcribe the following audio.",
                }
            }
        }
        mock_azure_openai_stt.return_value = "azure_openai_stt_instance"

        # Act
        result = get_azure_openai_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_openai_stt_instance")
        mock_azure_openai_stt.assert_called_once_with(
            api_key="test_azure_api_key",
            azure_ad_token="test_azure_ad_token",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
            azure_deployment="test_deployment",
            organization="test_org",
            project="test_project",
            language="es",
            detect_language=True,
            model="gpt-4o-transcribe",
            prompt="Transcribe the following audio.",
        )

    def test_get_azure_openai_stt_impl_missing_credentials(self):
        # Arrange
        config = {"live_captions": {"azure_openai": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_azure_openai_stt_impl(config)

        self.assertIn("Wrong Azure OpenAI credentials", str(context.exception))

    # Google STT Tests
    @patch("tempfile.NamedTemporaryFile")
    @patch("livekit.plugins.google.STT")
    def test_get_google_stt_impl_success(self, mock_google_stt, mock_temp_file):
        # Arrange
        config = {
            "live_captions": {
                "google": {
                    "credentials_info": '{"type": "service_account", "project_id": "test_project"}',
                    "model": "latest_short",
                    "languages": "fr-FR",
                    "detect_language": False,
                    "location": "europe-west1",
                    "punctuate": False,
                    "spoken_punctuation": True,
                    "interim_results": False,
                }
            }
        }

        # Mock the temp file
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_credentials.json"
        mock_temp_file.return_value = mock_file

        mock_google_stt.return_value = "google_stt_instance"

        # Act
        result = get_google_stt_impl(config)

        # Assert
        self.assertEqual(result, "google_stt_instance")
        mock_google_stt.assert_called_once_with(
            credentials_file="/tmp/test_credentials.json",
            model="latest_short",
            languages="fr-FR",
            detect_language=False,
            location="europe-west1",
            punctuate=False,
            spoken_punctuation=True,
            interim_results=False,
        )
        self.assertTrue(mock_file.write.called)
        self.assertTrue(mock_file.close.called)

    def test_get_google_stt_impl_missing_credentials(self):
        # Arrange
        config = {"live_captions": {"google": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_google_stt_impl(config)

        self.assertIn("Wrong Google credentials", str(context.exception))

    def test_get_google_stt_impl_invalid_json(self):
        # Arrange
        config = {"live_captions": {"google": {"credentials_info": "not_valid_json"}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_google_stt_impl(config)

        self.assertIn("must be a valid JSON", str(context.exception))

    @patch("tempfile.NamedTemporaryFile")
    def test_get_google_stt_impl_file_write_error(self, mock_temp_file):
        # Arrange
        config = {
            "live_captions": {
                "google": {"credentials_info": '{"type": "service_account"}'}
            }
        }

        # Make the temp file write fail
        mock_file = MagicMock()
        mock_file.write.side_effect = IOError("Write error")
        mock_temp_file.return_value = mock_file

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_google_stt_impl(config)

        self.assertIn("Failed to create a temporary JSON file", str(context.exception))

    # OpenAI STT Tests
    @patch("livekit.plugins.openai.STT")
    def test_get_openai_stt_impl_success(self, mock_openai_stt):
        # Arrange
        config = {
            "live_captions": {
                "openai": {
                    "api_key": "test_openai_key",
                    "model": "whisper-2",
                    "language": "fr",
                    "prompt": "<prompt>",
                    "detect_language": False,
                }
            }
        }
        mock_openai_stt.return_value = "openai_stt_instance"

        # Act
        result = get_openai_stt_impl(config)

        # Assert
        self.assertEqual(result, "openai_stt_instance")
        mock_openai_stt.assert_called_once_with(
            api_key="test_openai_key",
            model="whisper-2",
            language="fr",
            prompt="<prompt>",
            detect_language=False,
        )

    def test_get_openai_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"openai": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_openai_stt_impl(config)

        self.assertIn("Wrong OpenAI credentials", str(context.exception))

    # Groq STT Tests
    @patch("livekit.plugins.groq.STT")
    def test_get_groq_stt_impl_success(self, mock_groq_stt):
        # Arrange
        config = {
            "live_captions": {
                "groq": {
                    "api_key": "test_groq_key",
                    "model": "whisper-large-v3",
                    "language": "es",
                    "prompt": "You are a helpful assistant.",
                }
            }
        }
        mock_groq_stt.return_value = "groq_stt_instance"

        # Act
        result = get_groq_stt_impl(config)

        # Assert
        self.assertEqual(result, "groq_stt_instance")
        mock_groq_stt.assert_called_once_with(
            api_key="test_groq_key",
            model="whisper-large-v3",
            language="es",
            prompt="You are a helpful assistant.",
        )

    def test_get_groq_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"groq": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_groq_stt_impl(config)

        self.assertIn("Wrong Groq credentials", str(context.exception))

    @patch("livekit.plugins.deepgram.STT")  # Change this line
    def test_get_deepgram_stt_impl_success(self, mock_deepgram_stt):
        # Arrange
        config = {
            "live_captions": {
                "deepgram": {
                    "api_key": "test_deepgram_key",
                    "model": "nova-3",
                    "language": "en-US",
                    "detect_language": False,
                    "interim_results": False,
                    "smart_format": False,
                    "no_delay": True,
                    "punctuate": False,
                    "filler_words": False,
                    "profanity_filter": True,
                    "numerals": True,
                    "keywords": [["test1", 0.5], ["test2", 1.0]],
                    "keyterms": ["term1", "term2"],
                }
            }
        }
        mock_deepgram_stt.return_value = "deepgram_stt_instance"

        # Act
        result = get_deepgram_stt_impl(config)

        # Assert
        self.assertEqual(result, "deepgram_stt_instance")
        mock_deepgram_stt.assert_called_once_with(
            api_key="test_deepgram_key",
            model="nova-3",
            language="en-US",
            detect_language=False,
            interim_results=False,
            smart_format=False,
            no_delay=True,
            punctuate=False,
            filler_words=False,
            profanity_filter=True,
            numerals=True,
            keywords=[["test1", 0.5], ["test2", 1.0]],
            keyterms=["term1", "term2"],
        )

    def test_get_deepgram_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"deepgram": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_deepgram_stt_impl(config)

        self.assertIn("Wrong Deepgram credentials", str(context.exception))

    # AssemblyAI STT Tests
    @patch("livekit.plugins.assemblyai.STT")
    def test_get_assemblyai_stt_impl_success(self, mock_assemblyai_stt):
        # Arrange
        config = {
            "live_captions": {
                "assemblyai": {
                    "api_key": "test_assemblyai_key",
                    "end_of_turn_confidence_threshold": 0.1,
                    "min_end_of_turn_silence_when_confident": 120,
                    "max_turn_silence": 3000,
                    "format_turns": False,
                    "keyterms_prompt": ["term1", "term2"],
                }
            }
        }
        mock_assemblyai_stt.return_value = "assemblyai_stt_instance"

        # Act
        result = get_assemblyai_stt_impl(config)

        # Assert
        self.assertEqual(result, "assemblyai_stt_instance")
        mock_assemblyai_stt.assert_called_once_with(
            api_key="test_assemblyai_key",
            end_of_turn_confidence_threshold=0.1,
            min_end_of_turn_silence_when_confident=120,
            max_turn_silence=3000,
            format_turns=False,
            keyterms_prompt=["term1", "term2"],
        )

    def test_get_assemblyai_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"assemblyai": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_assemblyai_stt_impl(config)

        self.assertIn("Wrong AssemblyAI credentials", str(context.exception))

    # FAL STT Tests
    @patch("livekit.plugins.fal.WizperSTT")
    def test_get_fal_stt_impl_success(self, mock_fal_stt):
        # Arrange
        config = {
            "live_captions": {"fal": {"api_key": "test_fal_key", "language": "de"}}
        }
        mock_fal_stt.return_value = "fal_stt_instance"

        # Act
        result = get_fal_stt_impl(config)

        # Assert
        self.assertEqual(result, "fal_stt_instance")
        mock_fal_stt.assert_called_once_with(api_key="test_fal_key", language="de")

    def test_get_fal_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"fal": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_fal_stt_impl(config)

        self.assertIn("Wrong FAL credentials", str(context.exception))

    # Clova STT Tests
    @patch("livekit.plugins.clova.STT")
    def test_get_clova_stt_impl_success(self, mock_clova_stt):
        # Arrange
        config = {
            "live_captions": {
                "clova": {
                    "api_key": "test_clova_key",
                    "invoke_url": "https://test.api.clova.ai",
                    "language": "ko-KR",
                    "threshold": 0.7,
                }
            }
        }
        mock_clova_stt.return_value = "clova_stt_instance"

        # Act
        result = get_clova_stt_impl(config)

        # Assert
        self.assertEqual(result, "clova_stt_instance")
        mock_clova_stt.assert_called_once_with(
            invoke_url="https://test.api.clova.ai",
            secret="test_clova_key",
            language="ko-KR",
            threshold=0.7,
        )

    def test_get_clova_stt_impl_missing_api_key(self):
        # Arrange
        config = {
            "live_captions": {"clova": {"invoke_url": "https://test.api.clova.ai"}}
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_clova_stt_impl(config)

        self.assertIn("Wrong Clova credentials", str(context.exception))

    def test_get_clova_stt_impl_missing_invoke_url(self):
        # Arrange
        config = {"live_captions": {"clova": {"api_key": "test_clova_key"}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_clova_stt_impl(config)

        self.assertIn("Wrong Clova credentials", str(context.exception))

    # Speechmatics STT Tests
    @patch("livekit.plugins.speechmatics.STT")
    def test_get_speechmatics_stt_impl_success(self, mock_speechmatics_stt):
        # Arrange
        config = {
            "live_captions": {
                "speechmatics": {
                    "api_key": "test_speechmatics_key",
                    "language": "fr",
                    "model": "linden-1",
                    "operating_point": "enhanced",
                    "output_locale": "fr-FR",
                    "enable_partials": False,
                    "speaker_format": "{speaker_id}: {text}",
                    # retired by the Speechmatics Agent STT service; kept here so the
                    # assertions below prove they are no longer forwarded
                    "max_delay": 1.0,
                    "max_delay_mode": "fixed",
                    "punctuation_overrides": {"period": "full stop"},
                    "additional_vocab": [
                        {"content": "financial crisis"},
                        {
                            "content": "gnocchi",
                            "sounds_like": ["nyohki", "nokey", "nochi"],
                        },
                        {"content": "CEO", "sounds_like": ["C.E.O."]},
                    ],
                    "speaker_diarization_config": {
                        "max_speakers": 2,
                        "speaker_sensitivity": 0.5,
                        "prefer_current_speakers": True,
                    },
                }
            }
        }
        mock_speechmatics_stt.return_value = "speechmatics_stt_instance"

        # Act
        result = get_speechmatics_stt_impl(config)

        # Assert
        self.assertEqual(result, "speechmatics_stt_instance")
        mock_speechmatics_stt.assert_called_once()

        # Check the kwargs passed to STT constructor (individual parameters, not transcription_config)
        call_args = mock_speechmatics_stt.call_args[1]
        self.assertEqual(call_args["api_key"], "test_speechmatics_key")
        self.assertEqual(call_args["language"], "fr")
        self.assertEqual(call_args["model"], "linden-1")
        self.assertEqual(call_args["operating_point"], "enhanced")
        self.assertEqual(call_args["output_locale"], "fr-FR")
        self.assertEqual(call_args["speaker_format"], "{speaker_id}: {text}")
        self.assertEqual(
            call_args["additional_vocab"],
            [
                {"content": "financial crisis"},
                {"content": "gnocchi", "sounds_like": ["nyohki", "nokey", "nochi"]},
                {"content": "CEO", "sounds_like": ["C.E.O."]},
            ],
        )
        self.assertEqual(call_args["include_partials"], False)
        self.assertEqual(call_args["enable_diarization"], True)
        self.assertEqual(call_args["max_speakers"], 2)
        self.assertEqual(call_args["speaker_sensitivity"], 0.5)
        self.assertEqual(call_args["prefer_current_speaker"], True)
        # Retired by Speechmatics: forwarding them only makes the plugin log that it
        # ignores them, so they must not reach the constructor at all.
        for retired in ("max_delay", "max_delay_mode", "punctuation_overrides"):
            self.assertNotIn(retired, call_args)

    @patch("livekit.plugins.speechmatics.STT")
    def test_get_speechmatics_stt_impl_accepts_documented_diarization_key(
        self, mock_speechmatics_stt
    ):
        """agent-speech-processing.yaml documents the singular prefer_current_speaker.

        Only the plural was read until now, so a config written against the docs was
        silently ignored.
        """
        for key in ("prefer_current_speaker", "prefer_current_speakers"):
            with self.subTest(key=key):
                mock_speechmatics_stt.reset_mock()
                mock_speechmatics_stt.return_value = "speechmatics_stt_instance"
                config = {
                    "live_captions": {
                        "speechmatics": {
                            "api_key": "test_key",
                            "speaker_diarization_config": {key: True},
                        }
                    }
                }

                get_speechmatics_stt_impl(config)

                call_args = mock_speechmatics_stt.call_args[1]
                self.assertEqual(call_args["prefer_current_speaker"], True)

    @patch("stt_impl._get_cached_silero_vad")
    @patch("livekit.plugins.speechmatics.STT")
    def test_get_speechmatics_stt_impl_shares_the_cached_vad(
        self, mock_speechmatics_stt, mock_get_vad
    ):
        """The plugin must be handed the process-wide VAD, not left to load its own.

        Its default EXTERNAL turn detection closes turns from a VAD, and with none
        passed it loads a private Silero copy per STT, i.e. per participant.
        """
        # Arrange
        config = {"live_captions": {"speechmatics": {"api_key": "test_key"}}}
        sentinel_vad = object()
        mock_get_vad.return_value = sentinel_vad
        mock_speechmatics_stt.return_value = "speechmatics_stt_instance"

        # Act
        get_speechmatics_stt_impl(config)

        # Assert
        mock_get_vad.assert_called_once_with(load_if_missing=True)
        self.assertIs(mock_speechmatics_stt.call_args[1]["vad"], sentinel_vad)

    @patch("livekit.plugins.speechmatics.STT")
    def test_get_speechmatics_stt_impl_omits_unset_operating_point(
        self, mock_speechmatics_stt
    ):
        """An unset operating_point must not reach the plugin.

        livekit-plugins-speechmatics 1.8.2 turned `operating_point` into a deprecated
        alias of `model` that logs two warnings and resolves to the default model, so
        sending a default of our own would spam the log for a value nobody configured.
        """
        # Arrange
        config = {"live_captions": {"speechmatics": {"api_key": "test_key"}}}
        mock_speechmatics_stt.return_value = "speechmatics_stt_instance"

        # Act
        get_speechmatics_stt_impl(config)

        # Assert
        call_args = mock_speechmatics_stt.call_args[1]
        self.assertNotIn("operating_point", call_args)

    def test_get_speechmatics_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"speechmatics": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_speechmatics_stt_impl(config)

        self.assertIn("Wrong Speechmatics credentials", str(context.exception))

    # Gladia STT Tests
    @patch("livekit.plugins.gladia.STT")
    def test_get_gladia_stt_impl_success(self, mock_gladia_stt):
        # Arrange
        config = {
            "live_captions": {
                "gladia": {
                    "api_key": "test_gladia_key",
                    "languages": ["en"],
                    "interim_results": True,
                    "code_switching": True,
                }
            }
        }
        mock_gladia_stt.return_value = "gladia_stt_instance"

        # Act
        result = get_gladia_stt_impl(config)

        # Assert
        self.assertEqual(result, "gladia_stt_instance")
        mock_gladia_stt.assert_called_once_with(
            api_key="test_gladia_key",
            languages=["en"],
            interim_results=True,
            code_switching=True,
        )

    def test_get_gladia_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"gladia": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_gladia_stt_impl(config)

        self.assertIn("Wrong Gladia credentials", str(context.exception))

    # Sarvam STT Tests
    @patch("livekit.plugins.sarvam.STT")
    def test_get_sarvam_stt_impl_success(self, mock_sarvam_stt):
        # Arrange
        config = {
            "live_captions": {
                "sarvam": {
                    "api_key": "test_sarvam_key",
                    "language": "hi-IN",
                    "model": "saaras:v1",
                }
            }
        }
        mock_sarvam_stt.return_value = "sarvam_stt_instance"

        # Act
        result = get_sarvam_stt_impl(config)

        # Assert
        self.assertEqual(result, "sarvam_stt_instance")
        mock_sarvam_stt.assert_called_once_with(
            api_key="test_sarvam_key",
            language="hi-IN",
            model="saaras:v1",
        )

    def test_get_sarvam_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"sarvam": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_sarvam_stt_impl(config)

        self.assertIn("Wrong Sarvam credentials", str(context.exception))

    @patch("livekit.plugins.spitch.STT")
    def test_get_spitch_stt_impl_success(self, mock_spitch_stt):
        # Arrange
        config = {
            "live_captions": {
                "spitch": {
                    "api_key": "test_spitch_key",
                    "language": "de-DE",
                }
            }
        }
        mock_spitch_stt.return_value = "spitch_stt_instance"

        # Act
        result = get_spitch_stt_impl(config)

        # Assert
        self.assertEqual(result, "spitch_stt_instance")
        mock_spitch_stt.assert_called_once_with(
            language="de-DE",
        )
        # Check environment variable
        self.assertEqual(os.environ["SPITCH_API_KEY"], "test_spitch_key")

    @patch("livekit.plugins.spitch.STT")
    def test_get_spitch_stt_impl_default_language(self, mock_spitch_stt):
        # Arrange
        config = {
            "live_captions": {
                "spitch": {
                    "api_key": "test_spitch_key",
                    # No language specified - should use default
                }
            }
        }
        mock_spitch_stt.return_value = "spitch_stt_instance"

        # Act
        result = get_spitch_stt_impl(config)

        # Assert
        self.assertEqual(result, "spitch_stt_instance")
        mock_spitch_stt.assert_called_once_with(
            language="en",  # Default language
        )
        self.assertEqual(os.environ["SPITCH_API_KEY"], "test_spitch_key")

    def test_get_spitch_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"spitch": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_spitch_stt_impl(config)

        self.assertIn("Wrong Spitch credentials", str(context.exception))

    def test_get_spitch_stt_impl_none_api_key(self):
        # Arrange
        config = {"live_captions": {"spitch": {"api_key": None}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_spitch_stt_impl(config)

        self.assertIn("Wrong Spitch credentials", str(context.exception))

    # Cartesia STT Tests
    @patch("livekit.plugins.cartesia.STT")
    def test_get_cartesia_stt_impl_success(self, mock_cartesia_stt):
        # Arrange
        config = {
            "live_captions": {
                "cartesia": {
                    "api_key": "test_cartesia_key",
                    "language": "en",
                    "model": "sonic-english",
                }
            }
        }
        mock_cartesia_stt.return_value = "cartesia_stt_instance"

        # Act
        result = get_cartesia_stt_impl(config)

        # Assert
        self.assertEqual(result, "cartesia_stt_instance")
        mock_cartesia_stt.assert_called_once_with(
            api_key="test_cartesia_key",
            language="en",
            model="sonic-english",
        )

    def test_get_cartesia_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"cartesia": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_cartesia_stt_impl(config)

        self.assertIn("Wrong Cartesia credentials", str(context.exception))

    # Soniox STT Tests
    @patch("livekit.plugins.soniox.STT")
    def test_get_soniox_stt_impl_success(self, mock_soniox_stt):
        # Arrange
        config = {
            "live_captions": {
                "soniox": {
                    "api_key": "test_soniox_key",
                    "model": "premium",
                    "language_hints": ["en", "es"],
                    "context": "This is the context",
                }
            }
        }
        mock_soniox_stt.return_value = "soniox_stt_instance"

        # Act
        result = get_soniox_stt_impl(config)

        # Assert
        self.assertEqual(result, "soniox_stt_instance")
        mock_soniox_stt.assert_called_once()

        # Check the STTOptions
        call_args = mock_soniox_stt.call_args[1]
        config_obj = call_args["params"]

        mock_soniox_stt.assert_called_once_with(
            api_key="test_soniox_key",
            params=config_obj,
        )

        self.assertIsInstance(config_obj, SonioxSTTOptions)
        self.assertEqual(config_obj.model, "premium")
        self.assertEqual(config_obj.language_hints, ["en", "es"])
        self.assertEqual(config_obj.context, "This is the context")

    def test_get_soniox_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"soniox": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_soniox_stt_impl(config)

        self.assertIn("Wrong Soniox credentials", str(context.exception))

    # NVIDIA STT Tests
    @patch("livekit.plugins.nvidia.STT")
    def test_get_nvidia_stt_impl_success(self, mock_nvidia_stt):
        # Arrange
        config = {
            "live_captions": {
                "nvidia": {
                    "api_key": "test_nvidia_key",
                    "model": "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
                    "function_id": "1598d209-5e27-4d3c-8079-4751568b1081",
                    "punctuate": True,
                    "language_code": "en-US",
                    "sample_rate": 16000,
                    "server": "grpc.nvcf.nvidia.com:443",
                    "use_ssl": True,
                }
            }
        }
        mock_nvidia_stt.return_value = "nvidia_stt_instance"

        # Act
        result = get_nvidia_stt_impl(config)

        # Assert
        self.assertEqual(result, "nvidia_stt_instance")
        mock_nvidia_stt.assert_called_once_with(
            api_key="test_nvidia_key",
            model="parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
            function_id="1598d209-5e27-4d3c-8079-4751568b1081",
            punctuate=True,
            language_code="en-US",
            sample_rate=16000,
            server="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
        )

    @patch("livekit.plugins.nvidia.STT")
    def test_get_nvidia_stt_impl_minimal_config(self, mock_nvidia_stt):
        # Arrange - Only api_key provided
        config = {
            "live_captions": {
                "nvidia": {
                    "api_key": "test_nvidia_key",
                }
            }
        }
        mock_nvidia_stt.return_value = "nvidia_stt_instance"

        # Act
        result = get_nvidia_stt_impl(config)

        # Assert
        self.assertEqual(result, "nvidia_stt_instance")
        mock_nvidia_stt.assert_called_once_with(
            api_key="test_nvidia_key",
        )

    @patch("livekit.plugins.nvidia.STT")
    def test_get_nvidia_stt_impl_self_hosted_no_ssl(self, mock_nvidia_stt):
        # Arrange - Self-hosted NIM without SSL (api_key not required)
        config = {
            "live_captions": {
                "nvidia": {
                    "server": "localhost:50051",
                    "use_ssl": False,
                    "language_code": "es-ES",
                }
            }
        }
        mock_nvidia_stt.return_value = "nvidia_stt_instance"

        # Act
        result = get_nvidia_stt_impl(config)

        # Assert
        self.assertEqual(result, "nvidia_stt_instance")
        mock_nvidia_stt.assert_called_once_with(
            language_code="es-ES",
            server="localhost:50051",
            use_ssl=False,
        )

    def test_get_nvidia_stt_impl_missing_api_key_and_server(self):
        # Arrange - Neither api_key nor server provided
        config = {"live_captions": {"nvidia": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_nvidia_stt_impl(config)

        self.assertIn("Wrong NVIDIA configuration", str(context.exception))
        self.assertIn("api_key", str(context.exception))
        self.assertIn("server", str(context.exception))

    def test_get_stt_impl_nvidia(self):
        config = {"live_captions": {"provider": "nvidia"}}
        mock_impl = MagicMock(return_value="nvidia_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "nvidia": stt_impl.STT_PROVIDERS["nvidia"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "nvidia_stt_instance")
            mock_impl.assert_called_once_with(config)

    # ElevenLabs STT Tests
    @patch("livekit.plugins.elevenlabs.STT")
    def test_get_elevenlabs_stt_impl_success(self, mock_elevenlabs_stt):
        # Arrange
        config = {
            "live_captions": {
                "elevenlabs": {
                    "api_key": "test_elevenlabs_key",
                    "language_code": "en",
                    "model_id": "scribe_v1",
                    "base_url": "https://custom.api.elevenlabs.io",
                    "sample_rate": 16000,
                    "tag_audio_events": True,
                    "include_timestamps": False,
                }
            }
        }
        mock_elevenlabs_stt.return_value = "elevenlabs_stt_instance"

        # Act
        result = get_elevenlabs_stt_impl(config)

        # Assert
        self.assertEqual(result, "elevenlabs_stt_instance")
        mock_elevenlabs_stt.assert_called_once_with(
            api_key="test_elevenlabs_key",
            language_code="en",
            model_id="scribe_v1",
            base_url="https://custom.api.elevenlabs.io",
            sample_rate=16000,
            tag_audio_events=True,
            include_timestamps=False,
        )

    @patch("livekit.plugins.elevenlabs.STT")
    def test_get_elevenlabs_stt_impl_minimal_config(self, mock_elevenlabs_stt):
        # Arrange - Only api_key provided (mandatory)
        config = {
            "live_captions": {
                "elevenlabs": {
                    "api_key": "test_elevenlabs_key",
                }
            }
        }
        mock_elevenlabs_stt.return_value = "elevenlabs_stt_instance"

        # Act
        result = get_elevenlabs_stt_impl(config)

        # Assert
        self.assertEqual(result, "elevenlabs_stt_instance")
        mock_elevenlabs_stt.assert_called_once_with(
            api_key="test_elevenlabs_key",
        )

    def test_get_elevenlabs_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"elevenlabs": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_elevenlabs_stt_impl(config)

        self.assertIn("Wrong ElevenLabs credentials", str(context.exception))

    def test_get_elevenlabs_stt_impl_none_api_key(self):
        # Arrange
        config = {"live_captions": {"elevenlabs": {"api_key": None}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_elevenlabs_stt_impl(config)

        self.assertIn("Wrong ElevenLabs credentials", str(context.exception))

    def test_get_stt_impl_elevenlabs(self):
        config = {"live_captions": {"provider": "elevenlabs"}}
        mock_impl = MagicMock(return_value="elevenlabs_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "elevenlabs": stt_impl.STT_PROVIDERS["elevenlabs"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "elevenlabs_stt_instance")
            mock_impl.assert_called_once_with(config)

    # SimpliSmart STT Tests
    @patch("livekit.plugins.simplismart.STT")
    def test_get_simplismart_stt_impl_success(self, mock_simplismart_stt):
        # Arrange
        config = {
            "live_captions": {
                "simplismart": {
                    "api_key": "test_simplismart_key",
                    "model": "openai/whisper-large-v3-turbo",
                    "language": "en",
                    "task": "transcribe",
                    "without_timestamps": True,
                    "min_speech_duration_ms": 100,
                    "temperature": 0.0,
                    "multilingual": False,
                }
            }
        }
        mock_simplismart_stt.return_value = "simplismart_stt_instance"

        # Act
        result = get_simplismart_stt_impl(config)

        # Assert
        self.assertEqual(result, "simplismart_stt_instance")
        mock_simplismart_stt.assert_called_once_with(
            api_key="test_simplismart_key",
            model="openai/whisper-large-v3-turbo",
            language="en",
            task="transcribe",
            without_timestamps=True,
            min_speech_duration_ms=100,
            temperature=0.0,
            multilingual=False,
        )

    @patch("livekit.plugins.simplismart.STT")
    def test_get_simplismart_stt_impl_minimal_config(self, mock_simplismart_stt):
        # Arrange - Only api_key provided (mandatory)
        config = {
            "live_captions": {
                "simplismart": {
                    "api_key": "test_simplismart_key",
                }
            }
        }
        mock_simplismart_stt.return_value = "simplismart_stt_instance"

        # Act
        result = get_simplismart_stt_impl(config)

        # Assert
        self.assertEqual(result, "simplismart_stt_instance")
        mock_simplismart_stt.assert_called_once_with(
            api_key="test_simplismart_key",
        )

    def test_get_simplismart_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"simplismart": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_simplismart_stt_impl(config)

        self.assertIn("Wrong SimpliSmart credentials", str(context.exception))

    def test_get_simplismart_stt_impl_none_api_key(self):
        # Arrange
        config = {"live_captions": {"simplismart": {"api_key": None}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_simplismart_stt_impl(config)

        self.assertIn("Wrong SimpliSmart credentials", str(context.exception))

    def test_get_stt_impl_simplismart(self):
        config = {"live_captions": {"provider": "simplismart"}}
        mock_impl = MagicMock(return_value="simplismart_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "simplismart": stt_impl.STT_PROVIDERS["simplismart"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "simplismart_stt_instance")
            mock_impl.assert_called_once_with(config)

    # Master get_stt_impl Tests
    def test_get_stt_impl_missing_provider(self):
        # Arrange
        config = {"live_captions": {}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_stt_impl(config)

        self.assertIn("live_captions.provider not defined", str(context.exception))

    def test_get_stt_impl_no_speech_processing_section(self):
        # Arrange
        config = {}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_stt_impl(config)

        self.assertIn("live_captions.provider not defined", str(context.exception))

    def test_get_stt_impl_aws(self):
        # Arrange
        config = {"live_captions": {"provider": "aws"}}
        mock_impl = MagicMock(return_value="aws_stt_instance")

        # Patch the registry entry
        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {"aws": stt_impl.STT_PROVIDERS["aws"]._replace(impl_function=mock_impl)},
        ):
            # Act
            result = get_stt_impl(config)

            # Assert
            self.assertEqual(result, "aws_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_azure(self):
        config = {"live_captions": {"provider": "azure"}}
        mock_impl = MagicMock(return_value="azure_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "azure": stt_impl.STT_PROVIDERS["azure"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "azure_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_azure_openai(self):
        config = {"live_captions": {"provider": "azure_openai"}}
        mock_impl = MagicMock(return_value="azure_openai_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "azure_openai": stt_impl.STT_PROVIDERS["azure_openai"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "azure_openai_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_google(self):
        config = {"live_captions": {"provider": "google"}}
        mock_impl = MagicMock(return_value="google_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "google": stt_impl.STT_PROVIDERS["google"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "google_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_openai(self):
        config = {"live_captions": {"provider": "openai"}}
        mock_impl = MagicMock(return_value="openai_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "openai": stt_impl.STT_PROVIDERS["openai"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "openai_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_groq(self):
        config = {"live_captions": {"provider": "groq"}}
        mock_impl = MagicMock(return_value="groq_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {"groq": stt_impl.STT_PROVIDERS["groq"]._replace(impl_function=mock_impl)},
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "groq_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_deepgram(self):
        config = {"live_captions": {"provider": "deepgram"}}
        mock_impl = MagicMock(return_value="deepgram_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "deepgram": stt_impl.STT_PROVIDERS["deepgram"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "deepgram_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_assemblyai(self):
        config = {"live_captions": {"provider": "assemblyai"}}
        mock_impl = MagicMock(return_value="assemblyai_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "assemblyai": stt_impl.STT_PROVIDERS["assemblyai"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "assemblyai_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_fal(self):
        config = {"live_captions": {"provider": "fal"}}
        mock_impl = MagicMock(return_value="fal_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {"fal": stt_impl.STT_PROVIDERS["fal"]._replace(impl_function=mock_impl)},
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "fal_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_clova(self):
        config = {"live_captions": {"provider": "clova"}}
        mock_impl = MagicMock(return_value="clova_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "clova": stt_impl.STT_PROVIDERS["clova"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "clova_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_speechmatics(self):
        config = {"live_captions": {"provider": "speechmatics"}}
        mock_impl = MagicMock(return_value="speechmatics_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "speechmatics": stt_impl.STT_PROVIDERS["speechmatics"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "speechmatics_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_gladia(self):
        config = {"live_captions": {"provider": "gladia"}}
        mock_impl = MagicMock(return_value="gladia_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "gladia": stt_impl.STT_PROVIDERS["gladia"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "gladia_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_sarvam(self):
        config = {"live_captions": {"provider": "sarvam"}}
        mock_impl = MagicMock(return_value="sarvam_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "sarvam": stt_impl.STT_PROVIDERS["sarvam"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "sarvam_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_spitch(self):
        config = {"live_captions": {"provider": "spitch"}}
        mock_impl = MagicMock(return_value="spitch_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "spitch": stt_impl.STT_PROVIDERS["spitch"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "spitch_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_mistralai(self):
        config = {"live_captions": {"provider": "mistralai"}}
        mock_impl = MagicMock(return_value="mistralai_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "mistralai": stt_impl.STT_PROVIDERS["mistralai"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "mistralai_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_cartesia(self):
        config = {"live_captions": {"provider": "cartesia"}}
        mock_impl = MagicMock(return_value="cartesia_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "cartesia": stt_impl.STT_PROVIDERS["cartesia"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "cartesia_stt_instance")
            mock_impl.assert_called_once_with(config)

    def test_get_stt_impl_soniox(self):
        config = {"live_captions": {"provider": "soniox"}}
        mock_impl = MagicMock(return_value="soniox_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {
                "soniox": stt_impl.STT_PROVIDERS["soniox"]._replace(
                    impl_function=mock_impl
                )
            },
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "soniox_stt_instance")
            mock_impl.assert_called_once_with(config)

    # Vosk STT Tests
    @patch("livekit.plugins.vosk.STT")
    def test_get_vosk_stt_impl_success(self, mock_vosk_stt):
        # Arrange
        config = {
            "live_captions": {
                "vosk": {
                    "model": "vosk-model-small-es-0.42",
                    "language": "es",
                    "sample_rate": 16000,
                    "partial_results": True,
                }
            }
        }
        mock_vosk_stt.return_value = "vosk_stt_instance"

        # Act
        result = get_vosk_stt_impl(config)

        # Assert
        self.assertEqual(result, "vosk_stt_instance")
        mock_vosk_stt.assert_called_once_with(
            model_path="vosk-models/vosk-model-small-es-0.42",
            language="es",
            sample_rate=16000,
            partial_results=True,
        )

    @patch("livekit.plugins.vosk.STT")
    def test_get_vosk_stt_impl_auto_detect_language(self, mock_vosk_stt):
        # Arrange - No language provided, should auto-detect from model name
        config = {
            "live_captions": {
                "vosk": {
                    "model": "vosk-model-small-fr-0.22",
                }
            }
        }
        mock_vosk_stt.return_value = "vosk_stt_instance"

        # Act
        result = get_vosk_stt_impl(config)

        # Assert
        self.assertEqual(result, "vosk_stt_instance")
        mock_vosk_stt.assert_called_once_with(
            model_path="vosk-models/vosk-model-small-fr-0.22",
            language="fr",
        )

    @patch("livekit.plugins.vosk.STT")
    def test_get_vosk_stt_impl_unknown_model_defaults_to_en_us(self, mock_vosk_stt):
        # Arrange - Unknown model name, should default to en-US
        config = {
            "live_captions": {
                "vosk": {
                    "model": "vosk-model-unknown-language",
                }
            }
        }
        mock_vosk_stt.return_value = "vosk_stt_instance"

        # Act
        result = get_vosk_stt_impl(config)

        # Assert
        self.assertEqual(result, "vosk_stt_instance")
        mock_vosk_stt.assert_called_once_with(
            model_path="vosk-models/vosk-model-unknown-language",
            language="en-US",
        )

    @patch("livekit.plugins.vosk.STT")
    def test_get_vosk_stt_impl_no_model_defaults_to_en_us(self, mock_vosk_stt):
        # Arrange - No model or language, should default to en-US
        config = {"live_captions": {"vosk": {}}}
        mock_vosk_stt.return_value = "vosk_stt_instance"

        # Act
        result = get_vosk_stt_impl(config)

        # Assert
        self.assertEqual(result, "vosk_stt_instance")
        mock_vosk_stt.assert_called_once_with(
            language="en-US",
        )

    @patch("livekit.plugins.vosk.STT")
    def test_get_vosk_stt_impl_explicit_language_overrides_auto_detect(
        self, mock_vosk_stt
    ):
        # Arrange - Explicit language should override auto-detect
        config = {
            "live_captions": {
                "vosk": {
                    "model": "vosk-model-small-fr-0.22",
                    "language": "de",  # Explicit language different from model
                }
            }
        }
        mock_vosk_stt.return_value = "vosk_stt_instance"

        # Act
        result = get_vosk_stt_impl(config)

        # Assert
        self.assertEqual(result, "vosk_stt_instance")
        mock_vosk_stt.assert_called_once_with(
            model_path="vosk-models/vosk-model-small-fr-0.22",
            language="de",
        )

    def test_get_stt_impl_vosk(self):
        config = {"live_captions": {"provider": "vosk"}}
        mock_impl = MagicMock(return_value="vosk_stt_instance")

        with patch.dict(
            "stt_impl.STT_PROVIDERS",
            {"vosk": stt_impl.STT_PROVIDERS["vosk"]._replace(impl_function=mock_impl)},
        ):
            result = get_stt_impl(config)
            self.assertEqual(result, "vosk_stt_instance")
            mock_impl.assert_called_once_with(config)


class TestRetiredProviderKeys(unittest.TestCase):
    """stt_impl.RETIRED_PROVIDER_KEYS: the generic 'this key does nothing' mechanism.

    A provider's impl function stops reading a key its upstream service retired, so
    nothing is forwarded; the registry is what still tells the operator why their
    setting is being ignored, naming the live_captions.<provider>.<key> path rather
    than the plugin's own parameter.
    """

    def _config(self, provider: str, block: dict) -> dict:
        return {"live_captions": {"provider": provider, provider: block}}

    def test_registry_only_names_known_providers(self):
        for provider in stt_impl.RETIRED_PROVIDER_KEYS:
            with self.subTest(provider=provider):
                self.assertIn(
                    provider,
                    stt_impl.STT_PROVIDERS,
                    f"RETIRED_PROVIDER_KEYS names '{provider}', which is not a "
                    f"registered STT provider",
                )

    def test_every_entry_carries_a_reason(self):
        for provider, retired in stt_impl.RETIRED_PROVIDER_KEYS.items():
            for key, reason in retired.items():
                with self.subTest(provider=provider, key=key):
                    self.assertTrue(
                        isinstance(reason, str) and reason.strip(),
                        f"{provider}.{key} has no reason; the warning would read "
                        f"'is no longer supported: .'",
                    )

    def test_retired_keys_are_not_plugin_parameters(self):
        """A retired key that is a real constructor parameter again must be dropped.

        Checks the installed plugin, so a release that brings a parameter back fails
        here instead of leaving the agent warning about a key that works.
        """
        import importlib
        import inspect

        for provider, retired in stt_impl.RETIRED_PROVIDER_KEYS.items():
            config = stt_impl.STT_PROVIDERS[provider]
            try:
                module = importlib.import_module(config.plugin_module)
            except ModuleNotFoundError:
                self.skipTest(f"{config.plugin_module} not installed in this image")
            plugin_class = getattr(module, config.plugin_class)
            parameters = set(inspect.signature(plugin_class.__init__).parameters)
            for key in retired:
                with self.subTest(provider=provider, key=key):
                    self.assertNotIn(
                        key,
                        parameters,
                        f"{config.plugin_module}.{config.plugin_class} accepts "
                        f"'{key}' again: remove it from RETIRED_PROVIDER_KEYS and "
                        f"start reading it in get_{provider}_stt_impl.",
                    )

    def test_warning_names_the_openvidu_key(self):
        """The plugin names its own parameter; only this path finds it in the config."""
        config = self._config(
            "speechmatics",
            {"api_key": "k", "max_delay": 1.0, "max_delay_mode": "fixed"},
        )

        with self.assertLogs(level="WARNING") as captured:
            warned = stt_impl.warn_about_retired_keys(config, "speechmatics")

        logged = "\n".join(captured.output)
        self.assertCountEqual(warned, ["max_delay", "max_delay_mode"])
        self.assertIn("live_captions.speechmatics.max_delay", logged)
        self.assertIn("live_captions.speechmatics.max_delay_mode", logged)
        # punctuation_overrides is registered but not configured here
        self.assertNotIn("live_captions.speechmatics.punctuation_overrides", logged)

    def test_silent_when_the_keys_are_not_configured(self):
        config = self._config("speechmatics", {"api_key": "k"})

        with self.assertNoLogs(level="WARNING"):
            self.assertEqual(
                stt_impl.warn_about_retired_keys(config, "speechmatics"), []
            )

    def test_provider_without_retired_keys_is_a_no_op(self):
        config = self._config("deepgram", {"api_key": "k", "max_delay": 1.0})

        with self.assertNoLogs(level="WARNING"):
            self.assertEqual(stt_impl.warn_about_retired_keys(config, "deepgram"), [])

    def test_missing_or_malformed_provider_block_never_raises(self):
        for block in ({}, None, "not-a-mapping"):
            with self.subTest(block=block):
                config = {"live_captions": {"provider": "speechmatics"}}
                if block is not None:
                    config["live_captions"]["speechmatics"] = block
                self.assertEqual(
                    stt_impl.warn_about_retired_keys(config, "speechmatics"), []
                )

    @patch("stt_impl.plugin_is_available", return_value=True)
    def test_get_stt_impl_warns_for_any_provider(self, _mock_available):
        """Wired into the one dispatcher, so no provider can forget it."""
        config = self._config("speechmatics", {"api_key": "k", "max_delay": 1.0})

        with patch.dict(
            stt_impl.STT_PROVIDERS,
            {
                "speechmatics": stt_impl.STT_PROVIDERS["speechmatics"]._replace(
                    impl_function=MagicMock(return_value="stt_instance")
                )
            },
        ):
            with self.assertLogs(level="WARNING") as captured:
                result = stt_impl.get_stt_impl(config)

        self.assertEqual(result, "stt_instance")
        self.assertIn(
            "live_captions.speechmatics.max_delay", "\n".join(captured.output)
        )


class TestForwardedKwargsMatchInstalledPlugins(unittest.TestCase):
    """Every kwarg an impl function forwards must still be a parameter of its plugin.

    This is the generic form of what caught speechmatics: livekit-plugins-speechmatics
    1.8.2 dropped max_delay and punctuation_overrides, its STT absorbed them through
    **kwargs, and the agent went on sending values the service ignored. The other
    mocked tests cannot see that -- they assert against a MagicMock, which accepts
    anything -- so this one reads the real signature instead, for every provider at
    once, and fails on the next plugin release that retires a parameter.

    A key that legitimately disappears belongs in RETIRED_PROVIDER_KEYS, with the read
    removed from the impl function; then this test goes quiet again.
    """

    # impl functions that build their instance through a classmethod
    ALTERNATE_CONSTRUCTORS = {"azure_openai": ("with_azure",)}
    # impl functions whose kwargs feed a nested options object, not the constructor
    NESTED_OPTIONS = {"soniox": "STTOptions"}

    @staticmethod
    def _forwarded_kwargs(func: ast.FunctionDef) -> set[str]:
        """Keys of the `kwargs` dict the impl builds, plus any `kwargs[...] = ...`.

        Only those: they are unambiguously destined for the plugin. Keywords written
        directly at the call site are left out so the wrapper constructions some
        providers do (VADTriggeredSTT, StreamAdapter) are not mistaken for them.
        """
        names: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(t, ast.Name) and t.id == "kwargs" for t in node.targets):
                for inner in ast.walk(node.value):
                    if isinstance(inner, ast.Dict):
                        names |= {
                            k.value
                            for k in inner.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)
                        }
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "kwargs"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    names.add(target.slice.value)
        return names

    def _accepted_parameters(self, provider: str, config) -> set[str]:
        module = importlib.import_module(config.plugin_module)
        plugin_class = getattr(module, config.plugin_class)

        accepted = set(inspect.signature(plugin_class.__init__).parameters)
        for alternate in self.ALTERNATE_CONSTRUCTORS.get(provider, ()):
            accepted |= set(
                inspect.signature(getattr(plugin_class, alternate)).parameters
            )
        nested = self.NESTED_OPTIONS.get(provider)
        if nested and (options := getattr(module, nested, None)) is not None:
            accepted |= set(inspect.signature(options).parameters)
        return accepted

    def test_no_impl_forwards_an_unknown_kwarg(self):
        source = pathlib.Path(__file__).with_name("stt_impl.py").read_text()
        functions = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }

        checked = 0
        for provider, config in sorted(stt_impl.STT_PROVIDERS.items()):
            func = functions.get(f"get_{provider}_stt_impl")
            if func is None:
                continue
            with self.subTest(provider=provider):
                try:
                    accepted = self._accepted_parameters(provider, config)
                except ModuleNotFoundError:
                    self.skipTest(f"{config.plugin_module} not installed in this image")
                unknown = sorted(self._forwarded_kwargs(func) - accepted)
                self.assertEqual(
                    unknown,
                    [],
                    f"get_{provider}_stt_impl forwards {unknown} to "
                    f"{config.plugin_module}.{config.plugin_class}, which no longer "
                    f"declares them. If the plugin retired them, stop reading them "
                    f"and add them to RETIRED_PROVIDER_KEYS; if they were renamed, "
                    f"forward the new name.",
                )
                checked += 1

        self.assertGreater(checked, 0, "no provider was checked")


class TestDeprecatedPluginParameters(unittest.TestCase):
    """Catch a parameter the plugin starts calling deprecated while still declaring it.

    TestForwardedKwargsMatchInstalledPlugins only sees a parameter that *disappears*.
    The step before that -- still declared, but documented as deprecated and mapped to a
    new name -- is invisible to a signature check, and it is where the next break comes
    from: every key speechmatics retired in 1.8.2 spent a release in this state first.

    A new hit here is a decision, not a failure: migrate the config key to the new name,
    or record it below with the reason it is still fine.
    """

    # (provider, parameter) -> why forwarding it is still correct today.
    KNOWN_DEPRECATIONS = {
        ("assemblyai", "min_end_of_turn_silence_when_confident"):
            "deprecated alias, still mapped onto min_turn_silence by the plugin",
        ("deepgram", "keyterms"):
            "deprecated alias, still mapped onto keyterm by the plugin",
        ("elevenlabs", "model_id"):
            "deprecated alias, still mapped onto model by the plugin",
        ("speechmatics", "operating_point"):
            "deprecated alias of model, kept so existing configs keep working",
        ("speechmatics", "model"):
            "not deprecated: named in the operating_point deprecation notice as its "
            "replacement",
    }

    MARKER = re.compile(
        r"deprecat|no longer|is ignored|has no effect|will be removed", re.I
    )
    ALTERNATE_CONSTRUCTORS = (
        TestForwardedKwargsMatchInstalledPlugins.ALTERNATE_CONSTRUCTORS
    )

    def test_no_new_deprecation_among_forwarded_kwargs(self):
        source = pathlib.Path(__file__).with_name("stt_impl.py").read_text()
        functions = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }

        for provider, config in sorted(stt_impl.STT_PROVIDERS.items()):
            func = functions.get(f"get_{provider}_stt_impl")
            if func is None:
                continue
            with self.subTest(provider=provider):
                try:
                    module = importlib.import_module(config.plugin_module)
                except ModuleNotFoundError:
                    self.skipTest(f"{config.plugin_module} not installed in this image")
                plugin_class = getattr(module, config.plugin_class)

                # Only the constructor(s): a deprecation notice elsewhere in the package
                # (another method's docstring, the realtime model) is not about us.
                sources = [inspect.getsource(plugin_class.__init__)]
                for alternate in self.ALTERNATE_CONSTRUCTORS.get(provider, ()):
                    sources.append(inspect.getsource(getattr(plugin_class, alternate)))
                flagged = [
                    line.strip()
                    for text in sources
                    for line in text.splitlines()
                    if self.MARKER.search(line)
                ]

                forwarded = TestForwardedKwargsMatchInstalledPlugins._forwarded_kwargs(
                    func
                )
                for name in sorted(forwarded):
                    hits = [
                        line for line in flagged
                        if re.search(rf"\b{re.escape(name)}\b", line)
                    ]
                    if not hits:
                        continue
                    self.assertIn(
                        (provider, name),
                        self.KNOWN_DEPRECATIONS,
                        f"{config.plugin_module}.{config.plugin_class} now flags "
                        f"'{name}' as deprecated:\n    {hits[0]}\n"
                        f"Migrate live_captions.{provider}.{name} to the replacement, "
                        f"or add ('{provider}', '{name}') to KNOWN_DEPRECATIONS with "
                        f"the reason it is still correct.",
                    )


class TestConfigDocumentationMatchesCode(unittest.TestCase):
    """agent-speech-processing.yaml is the operator-facing contract; keep it honest.

    When a plugin retires a key, the impl stops reading it -- and without this test the
    YAML would go on documenting a setting that does nothing, which is exactly what an
    operator needs told. The reverse direction catches a key that works but was never
    written down.
    """

    # keys read on purpose without documenting them
    UNDOCUMENTED_ON_PURPOSE = {
        ("speechmatics", "operating_point"):
            "deprecated alias of model: still accepted so existing configs keep "
            "working, deliberately not advertised to new ones",
    }

    @classmethod
    def setUpClass(cls):
        import yaml

        base = pathlib.Path(__file__).parent
        cls.yaml_text = (base / "agent-speech-processing.yaml").read_text()
        cls.live_captions = yaml.safe_load(cls.yaml_text)["live_captions"]
        cls.functions = {
            node.name: node
            for node in ast.walk(ast.parse((base / "stt_impl.py").read_text()))
            if isinstance(node, ast.FunctionDef)
        }

    def _impl(self, provider):
        return self.functions.get(f"get_{provider}_stt_impl")

    def test_every_documented_key_is_read(self):
        """A documented key nobody reads silently does nothing."""
        for provider, block in sorted(self.live_captions.items()):
            if not isinstance(block, dict):
                continue
            func = self._impl(provider)
            if func is None:
                continue
            referenced = {
                node.value
                for node in ast.walk(func)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }

            def unread(prefix, node):
                for key, value in node.items():
                    if key not in referenced:
                        yield f"{prefix}{key}"
                    if isinstance(value, dict):
                        yield from unread(f"{prefix}{key}.", value)

            with self.subTest(provider=provider):
                stale = sorted(unread("", block))
                self.assertEqual(
                    stale,
                    [],
                    f"agent-speech-processing.yaml documents "
                    f"live_captions.{provider}.{{{', '.join(stale)}}}, which "
                    f"get_{provider}_stt_impl never reads. If the plugin retired the "
                    f"key, say so in the YAML and register it in "
                    f"RETIRED_PROVIDER_KEYS; otherwise start reading it.",
                )

    def test_every_read_key_is_documented(self):
        """A key that works but is written down nowhere is invisible to operators."""
        for provider in sorted(stt_impl.STT_PROVIDERS):
            func = self._impl(provider)
            if func is None:
                continue
            read = {
                node.args[0].value
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_value")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            }
            block = re.search(
                rf"^  {re.escape(provider)}:\n(.*?)(?=^  [a-z_]+:\n|\Z)",
                self.yaml_text,
                re.S | re.M,
            )
            documented_text = block.group(1) if block else ""

            with self.subTest(provider=provider):
                # commented-out examples count as documented, hence the raw text search
                missing = sorted(
                    key
                    for key in read
                    if not re.search(rf"\b{re.escape(key)}\b", documented_text)
                    and (provider, key) not in self.UNDOCUMENTED_ON_PURPOSE
                )
                self.assertEqual(
                    missing,
                    [],
                    f"get_{provider}_stt_impl reads {missing}, which "
                    f"agent-speech-processing.yaml never mentions. Document them, or "
                    f"record them in UNDOCUMENTED_ON_PURPOSE with the reason.",
                )
