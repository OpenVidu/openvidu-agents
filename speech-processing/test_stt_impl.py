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
    get_google_stt_impl,
    get_openai_stt_impl,
    get_groq_stt_impl,
    get_deepgram_stt_impl,
    get_assemblyai_stt_impl,
    get_fal_stt_impl,
    get_clova_stt_impl,
    get_speechmatics_stt_impl,
    get_stt_impl,
)


class TestSTTImplementations(unittest.TestCase):
    def setUp(self):
        # Set up common test data
        self.base_config = {"speech_processing": {"provider": "test_provider"}}

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
            "FAL_KEY",
            "SPEECHMATICS_API_KEY",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    # AWS STT Tests
    @patch("livekit.plugins.aws.STT")
    def test_get_aws_stt_impl_success(self, mock_aws_stt):
        # Arrange
        config = {
            "speech_processing": {
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
            api_key="test_key_id",
            api_secret="test_secret_key",
            speech_region="us-west-2",
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
        config = {"speech_processing": {"aws": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_aws_stt_impl(config)

        self.assertIn("Wrong AWS credentials", str(context.exception))

    def test_get_aws_stt_impl_partial_credentials(self):
        # Arrange
        config = {
            "speech_processing": {
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
            "speech_processing": {
                "azure": {
                    "speech_host": "test.host.com",
                    "languages": ["en-US", "es-ES"],
                }
            }
        }
        mock_azure_stt.return_value = "azure_stt_instance"

        # Act
        result = get_azure_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_azure_stt.assert_called_once_with(
            speech_host="test.host.com", languages=["en-US", "es-ES"]
        )

    @patch("livekit.plugins.azure.STT")
    def test_get_azure_stt_impl_with_key_region(self, mock_azure_stt):
        # Arrange
        config = {
            "speech_processing": {
                "azure": {
                    "speech_key": "test_key",
                    "speech_region": "westus",
                    "languages": ["en-US"],
                }
            }
        }
        mock_azure_stt.return_value = "azure_stt_instance"

        # Act
        result = get_azure_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_azure_stt.assert_called_once_with(
            speech_key="test_key", speech_region="westus", languages=["en-US"]
        )

    @patch("livekit.plugins.azure.STT")
    def test_get_azure_stt_impl_with_token_region(self, mock_azure_stt):
        # Arrange
        config = {
            "speech_processing": {
                "azure": {
                    "speech_auth_token": "test_token",
                    "speech_region": "westus",
                    "languages": ["en-US"],
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
            languages=["en-US"],
            profanity=ProfanityOption.Masked,
        )

    def test_get_azure_stt_impl_invalid_credentials(self):
        # Arrange
        config = {
            "speech_processing": {
                "azure": {
                    # Missing required credentials
                }
            }
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_azure_stt_impl(config)

        self.assertIn("Wrong azure credentials", str(context.exception))

    # Google STT Tests
    @patch("tempfile.NamedTemporaryFile")
    @patch("livekit.plugins.google.STT")
    def test_get_google_stt_impl_success(self, mock_google_stt, mock_temp_file):
        # Arrange
        config = {
            "speech_processing": {
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
        config = {"speech_processing": {"google": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_google_stt_impl(config)

        self.assertIn("Wrong Google credentials", str(context.exception))

    def test_get_google_stt_impl_invalid_json(self):
        # Arrange
        config = {"speech_processing": {"google": {"credentials_info": "not_valid_json"}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_google_stt_impl(config)

        self.assertIn("must be a valid JSON", str(context.exception))

    @patch("tempfile.NamedTemporaryFile")
    def test_get_google_stt_impl_file_write_error(self, mock_temp_file):
        # Arrange
        config = {
            "speech_processing": {
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
            "speech_processing": {
                "openai": {
                    "api_key": "test_openai_key",
                    "model": "whisper-2",
                    "language": "fr",
                }
            }
        }
        mock_openai_stt.return_value = "openai_stt_instance"

        # Act
        result = get_openai_stt_impl(config)

        # Assert
        self.assertEqual(result, "openai_stt_instance")
        mock_openai_stt.assert_called_once_with(
            api_key="test_openai_key", model="whisper-2", language="fr"
        )

    def test_get_openai_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"openai": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_openai_stt_impl(config)

        self.assertIn("Wrong OpenAI credentials", str(context.exception))

    # Groq STT Tests
    @patch("livekit.plugins.openai.stt.STT.with_groq")
    def test_get_groq_stt_impl_success(self, mock_groq_stt):
        # Arrange
        config = {
            "speech_processing": {
                "groq": {
                    "api_key": "test_groq_key",
                    "model": "whisper-large-v3",
                    "language": "es",
                }
            }
        }
        mock_groq_stt.return_value = "groq_stt_instance"

        # Act
        result = get_groq_stt_impl(config)

        # Assert
        self.assertEqual(result, "groq_stt_instance")
        mock_groq_stt.assert_called_once_with(
            api_key="test_groq_key", model="whisper-large-v3", language="es"
        )

    def test_get_groq_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"groq": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_groq_stt_impl(config)

        self.assertIn("Wrong Groq credentials", str(context.exception))

    # Deepgram STT Tests
    @patch("livekit.plugins.deepgram.stt.STT")
    def test_get_deepgram_stt_impl_success(self, mock_deepgram_stt):
        # Arrange
        config = {
            "speech_processing": {
                "deepgram": {
                    "api_key": "test_deepgram_key",
                    "model": "nova-2-meeting",
                    "language": "en-GB",
                    "interim_results": False,
                    "smart_format": False,
                    "punctuate": False,
                    "filler_words": False,
                    "profanity_filter": True,
                    "keywords": ["test", "keywords"],
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
            model="nova-2-meeting",
            language="en-GB",
            interim_results=False,
            smart_format=False,
            punctuate=False,
            filler_words=False,
            profanity_filter=True,
            keywords=["test", "keywords"],
            keyterms=["term1", "term2"],
        )

    def test_get_deepgram_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"deepgram": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_deepgram_stt_impl(config)

        self.assertIn("Wrong Deepgram credentials", str(context.exception))

    # AssemblyAI STT Tests
    @patch("livekit.plugins.assemblyai.STT")
    def test_get_assemblyai_stt_impl_success(self, mock_assemblyai_stt):
        # Arrange
        config = {
            "speech_processing": {
                "assemblyai": {
                    "api_key": "test_assemblyai_key",
                    "word_boost": ["boost", "these", "words"],
                    "disable_partial_transcripts": True,
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
            word_boost=["boost", "these", "words"],
            disable_partial_transcripts=True,
        )

    def test_get_assemblyai_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"assemblyai": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_assemblyai_stt_impl(config)

        self.assertIn("Wrong AssemblyAI credentials", str(context.exception))

    # FAL STT Tests
    @patch("livekit.plugins.fal.WizperSTT")
    def test_get_fal_stt_impl_success(self, mock_fal_stt):
        # Arrange
        config = {
            "speech_processing": {
                "fal": {
                    "api_key": "test_fal_key",
                    "task": "translate",
                    "language": "de",
                    "chunk_level": "word",
                    "version": "2",
                }
            }
        }
        mock_fal_stt.return_value = "fal_stt_instance"

        # Act
        result = get_fal_stt_impl(config)

        # Assert
        self.assertEqual(result, "fal_stt_instance")
        mock_fal_stt.assert_called_once_with(
            task="translate", language="de", chunk_level="word", version="2"
        )
        self.assertEqual(os.environ["FAL_KEY"], "test_fal_key")

    def test_get_fal_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"fal": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_fal_stt_impl(config)

        self.assertIn("Wrong FAL credentials", str(context.exception))

    # Clova STT Tests
    @patch("livekit.plugins.clova.STT")
    def test_get_clova_stt_impl_success(self, mock_clova_stt):
        # Arrange
        config = {
            "speech_processing": {
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
            "speech_processing": {"clova": {"invoke_url": "https://test.api.clova.ai"}}
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_clova_stt_impl(config)

        self.assertIn("Wrong Clova credentials", str(context.exception))

    def test_get_clova_stt_impl_missing_invoke_url(self):
        # Arrange
        config = {"speech_processing": {"clova": {"api_key": "test_clova_key"}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_clova_stt_impl(config)

        self.assertIn("Wrong Clova credentials", str(context.exception))

    # Speechmatics STT Tests
    @patch("livekit.plugins.speechmatics.STT")
    def test_get_speechmatics_stt_impl_success(self, mock_speechmatics_stt):
        # Arrange
        config = {
            "speech_processing": {
                "speechmatics": {
                    "api_key": "test_speechmatics_key",
                    "language": "fr",
                    "output_locale": "fr-FR",
                    "enable_partials": False,
                    "max_delay": 1.0,
                    "max_delay_mode": "fixed",
                    "punctuation_overrides": {"period": "full stop"},
                    "additional_vocab": ["vocabulary", "additions"],
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
        self.assertEqual(config_obj.additional_vocab, ["vocabulary", "additions"])
        self.assertEqual(config_obj.enable_partials, False)
        self.assertEqual(config_obj.max_delay, 1.0)
        self.assertEqual(config_obj.max_delay_mode, "fixed")

        # Check environment variable
        self.assertEqual(os.environ["SPEECHMATICS_API_KEY"], "test_speechmatics_key")

    def test_get_speechmatics_stt_impl_missing_api_key(self):
        # Arrange
        config = {"speech_processing": {"speechmatics": {}}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_speechmatics_stt_impl(config)

        self.assertIn("Wrong Speechmatics credentials", str(context.exception))

    # Master get_stt_impl Tests
    def test_get_stt_impl_missing_provider(self):
        # Arrange
        config = {"speech_processing": {}}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_stt_impl(config)

        self.assertIn("speech_processing.provider not defined", str(context.exception))

    def test_get_stt_impl_no_speech_processing_section(self):
        # Arrange
        config = {}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_stt_impl(config)

        self.assertIn("speech_processing.provider not defined", str(context.exception))

    @patch("stt_impl.get_aws_stt_impl")
    def test_get_stt_impl_aws(self, mock_get_aws):
        # Arrange
        config = {"speech_processing": {"provider": "aws"}}
        mock_get_aws.return_value = "aws_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "aws_stt_instance")
        mock_get_aws.assert_called_once_with(config)

    @patch("stt_impl.get_azure_stt_impl")
    def test_get_stt_impl_azure(self, mock_get_azure):
        # Arrange
        config = {"speech_processing": {"provider": "azure"}}
        mock_get_azure.return_value = "azure_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "azure_stt_instance")
        mock_get_azure.assert_called_once_with(config)

    @patch("stt_impl.get_google_stt_impl")
    def test_get_stt_impl_google(self, mock_get_google):
        # Arrange
        config = {"speech_processing": {"provider": "google"}}
        mock_get_google.return_value = "google_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "google_stt_instance")
        mock_get_google.assert_called_once_with(config)

    @patch("stt_impl.get_openai_stt_impl")
    def test_get_stt_impl_openai(self, mock_get_openai):
        # Arrange
        config = {"speech_processing": {"provider": "openai"}}
        mock_get_openai.return_value = "openai_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "openai_stt_instance")
        mock_get_openai.assert_called_once_with(config)

    @patch("stt_impl.get_groq_stt_impl")
    def test_get_stt_impl_groq(self, mock_get_groq):
        # Arrange
        config = {"speech_processing": {"provider": "groq"}}
        mock_get_groq.return_value = "groq_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "groq_stt_instance")
        mock_get_groq.assert_called_once_with(config)

    @patch("stt_impl.get_deepgram_stt_impl")
    def test_get_stt_impl_deepgram(self, mock_get_deepgram):
        # Arrange
        config = {"speech_processing": {"provider": "deepgram"}}
        mock_get_deepgram.return_value = "deepgram_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "deepgram_stt_instance")
        mock_get_deepgram.assert_called_once_with(config)

    @patch("stt_impl.get_assemblyai_stt_impl")
    def test_get_stt_impl_assemblyai(self, mock_get_assemblyai):
        # Arrange
        config = {"speech_processing": {"provider": "assemblyai"}}
        mock_get_assemblyai.return_value = "assemblyai_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "assemblyai_stt_instance")
        mock_get_assemblyai.assert_called_once_with(config)

    @patch("stt_impl.get_fal_stt_impl")
    def test_get_stt_impl_fal(self, mock_get_fal):
        # Arrange
        config = {"speech_processing": {"provider": "fal"}}
        mock_get_fal.return_value = "fal_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "fal_stt_instance")
        mock_get_fal.assert_called_once_with(config)

    @patch("stt_impl.get_clova_stt_impl")
    def test_get_stt_impl_clova(self, mock_get_clova):
        # Arrange
        config = {"speech_processing": {"provider": "clova"}}
        mock_get_clova.return_value = "clova_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "clova_stt_instance")
        mock_get_clova.assert_called_once_with(config)

    @patch("stt_impl.get_speechmatics_stt_impl")
    def test_get_stt_impl_speechmatics(self, mock_get_speechmatics):
        # Arrange
        config = {"speech_processing": {"provider": "speechmatics"}}
        mock_get_speechmatics.return_value = "speechmatics_stt_instance"

        # Act
        result = get_stt_impl(config)

        # Assert
        self.assertEqual(result, "speechmatics_stt_instance")
        mock_get_speechmatics.assert_called_once_with(config)
