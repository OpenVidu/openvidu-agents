import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import logging
from io import StringIO

# Import the module to test
import sys

sys.path.append(".")  # Adjust as needed for your project structure
from azure.cognitiveservices.speech.enums import ProfanityOption
from livekit.plugins.speechmatics.types import TranscriptionConfig

# Import the module containing the functions we want to test
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
                    "operating_point": "enhanced",
                    "output_locale": "fr-FR",
                    "enable_partials": False,
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

        # Check the TranscriptionConfig
        call_args = mock_speechmatics_stt.call_args[1]
        config_obj = call_args["transcription_config"]
        self.assertIsInstance(config_obj, TranscriptionConfig)
        self.assertEqual(config_obj.operating_point, "enhanced")
        self.assertEqual(config_obj.language, "fr")
        self.assertEqual(config_obj.output_locale, "fr-FR")
        self.assertEqual(config_obj.punctuation_overrides, {"period": "full stop"})
        self.assertEqual(
            config_obj.additional_vocab,
            [
                {"content": "financial crisis"},
                {"content": "gnocchi", "sounds_like": ["nyohki", "nokey", "nochi"]},
                {"content": "CEO", "sounds_like": ["C.E.O."]},
            ],
        )
        self.assertEqual(config_obj.enable_partials, False)
        self.assertEqual(config_obj.max_delay, 1.0)
        self.assertEqual(config_obj.max_delay_mode, "fixed")
        self.assertEqual(config_obj.speaker_diarization_config["max_speakers"], 2)
        self.assertEqual(
            config_obj.speaker_diarization_config["speaker_sensitivity"], 0.5
        )
        self.assertEqual(
            config_obj.speaker_diarization_config["prefer_current_speakers"], True
        )

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
        mock_soniox_stt.assert_called_once_with(
            api_key="test_soniox_key",
            model="premium",
            language_hints=["en", "es"],
            context="This is the context",
        )

    def test_get_soniox_stt_impl_missing_api_key(self):
        # Arrange
        config = {"live_captions": {"soniox": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_soniox_stt_impl(config)

        self.assertIn("Wrong Soniox credentials", str(context.exception))

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

    @patch("stt_impl.get_aws_stt_impl")
    def test_get_stt_impl_aws(self, mock_get_aws):
        # Arrange
        config = {"live_captions": {"provider": "aws"}}
        mock_get_aws.return_value = "aws_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "aws_stt_instance")
        mock_get_aws.assert_called_once_with(config)

    @patch("stt_impl.get_azure_stt_impl")
    def test_get_stt_impl_azure(self, mock_get_azure):
        # Arrange
        config = {"live_captions": {"provider": "azure"}}
        mock_get_azure.return_value = "azure_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_get_azure.assert_called_once_with(config)

    @patch("stt_impl.get_azure_openai_stt_impl")
    def test_get_stt_impl_azure_openai(self, mock_get_azure_openai):
        # Arrange
        config = {"live_captions": {"provider": "azure_openai"}}
        mock_get_azure_openai.return_value = "azure_openai_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_openai_stt_instance")
        mock_get_azure_openai.assert_called_once_with(config)

    @patch("stt_impl.get_google_stt_impl")
    def test_get_stt_impl_google(self, mock_get_google):
        # Arrange
        config = {"live_captions": {"provider": "google"}}
        mock_get_google.return_value = "google_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "google_stt_instance")
        mock_get_google.assert_called_once_with(config)

    @patch("stt_impl.get_openai_stt_impl")
    def test_get_stt_impl_openai(self, mock_get_openai):
        # Arrange
        config = {"live_captions": {"provider": "openai"}}
        mock_get_openai.return_value = "openai_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "openai_stt_instance")
        mock_get_openai.assert_called_once_with(config)

    @patch("stt_impl.get_groq_stt_impl")
    def test_get_stt_impl_groq(self, mock_get_groq):
        # Arrange
        config = {"live_captions": {"provider": "groq"}}
        mock_get_groq.return_value = "groq_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "groq_stt_instance")
        mock_get_groq.assert_called_once_with(config)

    @patch("stt_impl.get_deepgram_stt_impl")
    def test_get_stt_impl_deepgram(self, mock_get_deepgram):
        # Arrange
        config = {"live_captions": {"provider": "deepgram"}}
        mock_get_deepgram.return_value = "deepgram_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "deepgram_stt_instance")
        mock_get_deepgram.assert_called_once_with(config)

    @patch("stt_impl.get_assemblyai_stt_impl")
    def test_get_stt_impl_assemblyai(self, mock_get_assemblyai):
        # Arrange
        config = {"live_captions": {"provider": "assemblyai"}}
        mock_get_assemblyai.return_value = "assemblyai_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "assemblyai_stt_instance")
        mock_get_assemblyai.assert_called_once_with(config)

    @patch("stt_impl.get_fal_stt_impl")
    def test_get_stt_impl_fal(self, mock_get_fal):
        # Arrange
        config = {"live_captions": {"provider": "fal"}}
        mock_get_fal.return_value = "fal_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "fal_stt_instance")
        mock_get_fal.assert_called_once_with(config)

    @patch("stt_impl.get_clova_stt_impl")
    def test_get_stt_impl_clova(self, mock_get_clova):
        # Arrange
        config = {"live_captions": {"provider": "clova"}}
        mock_get_clova.return_value = "clova_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "clova_stt_instance")
        mock_get_clova.assert_called_once_with(config)

    @patch("stt_impl.get_speechmatics_stt_impl")
    def test_get_stt_impl_speechmatics(self, mock_get_speechmatics):
        # Arrange
        config = {"live_captions": {"provider": "speechmatics"}}
        mock_get_speechmatics.return_value = "speechmatics_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "speechmatics_stt_instance")
        mock_get_speechmatics.assert_called_once_with(config)

    @patch("stt_impl.get_gladia_stt_impl")
    def test_get_stt_impl_gladia(self, mock_get_gladia):
        # Arrange
        config = {"live_captions": {"provider": "gladia"}}
        mock_get_gladia.return_value = "gladia_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "gladia_stt_instance")
        mock_get_gladia.assert_called_once_with(config)

    @patch("stt_impl.get_sarvam_stt_impl")
    def test_get_stt_impl_sarvam(self, mock_get_sarvam):
        # Arrange
        config = {"live_captions": {"provider": "sarvam"}}
        mock_get_sarvam.return_value = "sarvam_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "sarvam_stt_instance")
        mock_get_sarvam.assert_called_once_with(config)

    @patch("stt_impl.get_spitch_stt_impl")
    def test_get_stt_impl_spitch(self, mock_get_spitch):
        # Arrange
        config = {"live_captions": {"provider": "spitch"}}
        mock_get_spitch.return_value = "spitch_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "spitch_stt_instance")
        mock_get_spitch.assert_called_once_with(config)

    @patch("stt_impl.get_cartesia_stt_impl")
    def test_get_stt_impl_cartesia(self, mock_get_cartesia):
        # Arrange
        config = {"live_captions": {"provider": "cartesia"}}
        mock_get_cartesia.return_value = "cartesia_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "cartesia_stt_instance")
        mock_get_cartesia.assert_called_once_with(config)

    @patch("stt_impl.get_soniox_stt_impl")
    def test_get_stt_impl_soniox(self, mock_get_soniox):
        # Arrange
        config = {"live_captions": {"provider": "soniox"}}
        mock_get_soniox.return_value = "soniox_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "soniox_stt_instance")
        mock_get_soniox.assert_called_once_with(config)
