import unittest
import os
from unittest.mock import patch

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
    get_stt_impl,
)


def generate_agent_config(provider, config):
    return {"speech_to_text": {"provider": provider, provider: config}}


class TestSTTProviders(unittest.TestCase):
    @patch("stt_impl.aws.STT")
    def test_get_aws_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "aws",
            {
                "aws_access_key_id": "KEY",
                "aws_secret_access_key": "SECRET",
                "aws_default_region": "us-east-1",
            },
        )
        get_aws_stt_impl(agent_config)
        mock_stt.assert_called_once_with(
            api_key="KEY", api_secret="SECRET", speech_region="us-east-1"
        )

    def test_get_aws_stt_impl_invalid(self):
        agent_config = generate_agent_config("aws", {})
        with self.assertRaises(ValueError) as ctx:
            get_aws_stt_impl(agent_config)
        self.assertIn("Wrong AWS credentials", str(ctx.exception))
        agent_config = generate_agent_config("aws", {"aws_access_key_id": "KEY"})
        with self.assertRaises(ValueError) as ctx:
            get_aws_stt_impl(agent_config)
        self.assertIn("Wrong AWS credentials", str(ctx.exception))
        agent_config = generate_agent_config(
            "aws", {"aws_access_key_id": "KEY", "aws_secret_access_key": "SECRET"}
        )
        with self.assertRaises(ValueError) as ctx:
            get_aws_stt_impl(agent_config)
        self.assertIn("Wrong AWS credentials", str(ctx.exception))

    @patch("stt_impl.azure.STT")
    def test_get_azure_stt_impl_valid_speech_host(self, mock_stt):
        agent_config = generate_agent_config("azure", {"speech_host": "some_host"})
        get_azure_stt_impl(agent_config)
        mock_stt.assert_called_once_with(speech_host="some_host")

    def test_get_azure_stt_impl_invalid_config(self):
        agent_config = generate_agent_config("azure", {})
        with self.assertRaises(ValueError) as ctx:
            get_azure_stt_impl(agent_config)
        self.assertIn("Wrong azure credentials", str(ctx.exception))

    @patch("stt_impl.google.STT")
    def test_get_google_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "google", {"credentials_file": "file.json"}
        )
        get_google_stt_impl(agent_config)
        mock_stt.assert_called_once_with(
            model="chirp_2", spoken_punctuation=True, credentials_file="file.json"
        )

    def test_get_google_stt_impl_invalid(self):
        agent_config = generate_agent_config("google", {})
        with self.assertRaises(ValueError) as ctx:
            get_google_stt_impl(agent_config)
        self.assertIn("Wrong Google credentials", str(ctx.exception))

    @patch("stt_impl.openai.STT")
    def test_get_openai_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config("openai", {"api_key": "key123"})
        get_openai_stt_impl(agent_config)
        mock_stt.assert_called_once_with(api_key="key123")

    def test_get_openai_stt_impl_invalid(self):
        agent_config = generate_agent_config("openai", {})
        with self.assertRaises(ValueError) as ctx:
            get_openai_stt_impl(agent_config)
        self.assertIn("Wrong OpenAI credentials", str(ctx.exception))

    @patch("stt_impl.openai.stt.STT.with_groq")
    def test_get_groq_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "groq", {"api_key": "key123", "model": None, "language": None}
        )
        get_groq_stt_impl(agent_config)
        mock_stt.assert_called_once_with(
            api_key="key123", model="whisper-large-v3-turbo", language="en"
        )

    def test_get_groq_stt_impl_invalid(self):
        agent_config = generate_agent_config("groq", {})
        with self.assertRaises(ValueError) as ctx:
            get_groq_stt_impl(agent_config)
        self.assertIn("Wrong Groq credentials", str(ctx.exception))

    @patch("stt_impl.deepgram.stt.STT")
    def test_get_deepgram_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "deepgram", {"api_key": "key123", "model": None, "language": None}
        )
        get_deepgram_stt_impl(agent_config)
        mock_stt.assert_called_once_with(
            model="nova-2-general",
            language="en-US",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            filler_words=True,
            profanity_filter=False,
            keywords=[("LiveKit", 1.5)],
        )

    def test_get_deepgram_stt_impl_invalid(self):
        agent_config = generate_agent_config("deepgram", {})
        with self.assertRaises(ValueError) as ctx:
            get_deepgram_stt_impl(agent_config)
        self.assertIn("Wrong Deepgram credentials", str(ctx.exception))

    @patch("stt_impl.assemblyai.STT")
    def test_get_assemblyai_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config("assemblyai", {"api_key": "key123"})
        get_assemblyai_stt_impl(agent_config)
        mock_stt.assert_called_once_with(api_key="key123")

    def test_get_assemblyai_stt_impl_invalid(self):
        agent_config = generate_agent_config("assemblyai", {})
        with self.assertRaises(ValueError) as ctx:
            get_assemblyai_stt_impl(agent_config)
        self.assertIn("Wrong AssemblyAI credentials", str(ctx.exception))

    @patch("stt_impl.fal.WizperSTT")
    @patch("os.environ", {})
    def test_get_fal_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "fal", {"api_key": "key123", "language": "en"}
        )
        get_fal_stt_impl(agent_config)
        self.assertEqual(os.environ["FAL_KEY"], "key123")
        mock_stt.assert_called_once_with(language="en")

    def test_get_fal_stt_impl_invalid(self):
        agent_config = generate_agent_config("fal", {})
        with self.assertRaises(ValueError) as ctx:
            get_fal_stt_impl(agent_config)
        self.assertIn("Wrong FAL credentials", str(ctx.exception))

    @patch("stt_impl.clova.STT")
    def test_get_clova_stt_impl_valid(self, mock_stt):
        agent_config = generate_agent_config(
            "clova", {"api_key": "key123", "invoke_url": "url"}
        )
        get_clova_stt_impl(agent_config)
        mock_stt.assert_called_once_with(invoke_url="url", secret="key123")

    def test_get_clova_stt_impl_invalid(self):
        agent_config = generate_agent_config("clova", {})
        with self.assertRaises(ValueError) as ctx:
            get_clova_stt_impl(agent_config)
        self.assertIn("Wrong Clova credentials", str(ctx.exception))

    @patch("stt_impl.get_azure_stt_impl")
    @patch("stt_impl.get_google_stt_impl")
    @patch("stt_impl.get_openai_stt_impl")
    def test_get_stt_impl_valid(self, mock_openai, mock_google, mock_azure):
        mock_azure.return_value = "azure_stt_instance"
        mock_google.return_value = "google_stt_instance"
        mock_openai.return_value = "openai_stt_instance"

        agent_config = generate_agent_config("azure", {"speech_host": "some_host"})
        self.assertEqual(get_stt_impl(agent_config), "azure_stt_instance")

        agent_config = generate_agent_config(
            "google", {"credentials_file": "file.json"}
        )
        self.assertEqual(get_stt_impl(agent_config), "google_stt_instance")

        agent_config = generate_agent_config("openai", {"api_key": "key123"})
        self.assertEqual(get_stt_impl(agent_config), "openai_stt_instance")

    def test_get_stt_impl_invalid_provider(self):
        agent_config = generate_agent_config("invalid", {})
        with self.assertRaises(ValueError) as ctx:
            get_stt_impl(agent_config)
        self.assertIn("unknown STT provider", str(ctx.exception))

    def test_get_stt_impl_missing_provider(self):
        agent_config = {"speech_to_text": {}}
        with self.assertRaises(ValueError) as ctx:
            get_stt_impl(agent_config)
        self.assertIn("speech_to_text.provider not defined", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
