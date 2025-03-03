import unittest
import os
import tempfile
from unittest.mock import patch
from transcriber import load_agent_config, is_agent_config_file


class TestLoadAgentConfig(unittest.TestCase):
    def setUp(self):
        self.logger_patch = patch("transcriber.logger")
        self.mock_logger = self.logger_patch.start()

    def tearDown(self):
        self.logger_patch.stop()
        os.environ.pop("AGENT_CONFIG_BODY", None)
        os.environ.pop("AGENT_CONFIG_FILE", None)
        os.environ.pop("LIVEKIT_API_KEY", None)
        os.environ.pop("LIVEKIT_API_SECRET", None)
        os.environ.pop("LIVEKIT_URL", None)

    def test_load_config_from_env_var_body(self):
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: test-url
        """
        os.environ["AGENT_CONFIG_BODY"] = config_yaml

        agent_config, agent_name = load_agent_config()
        self.assertEqual(agent_name, "test-agent")
        self.assertEqual(agent_config["api_key"], "test-key")
        self.assertEqual(agent_config["api_secret"], "test-secret")
        self.assertEqual(agent_config["ws_url"], "test-url")

    def test_load_config_invalid_yaml_env_var_body(self):
        os.environ["AGENT_CONFIG_BODY"] = "invalid: yaml: ::"

        with self.assertRaises(SystemExit):
            load_agent_config()
        self.mock_logger.error.assert_called()

    def test_missing_agent_name_in_env_var_body(self):
        config_yaml = """
        api_key: test-key
        api_secret: test-secret
        ws_url: test-url
        """
        os.environ["AGENT_CONFIG_BODY"] = config_yaml

        with self.assertRaises(SystemExit):
            load_agent_config()
        self.mock_logger.error.assert_called_with(
            'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
        )

    def test_blank_agent_name_in_env_var_body(self):
        config_yaml = """
        agent_name:
        api_key: test-key
        api_secret: test-secret
        ws_url: test-url
        """
        os.environ["AGENT_CONFIG_BODY"] = config_yaml

        with self.assertRaises(SystemExit):
            load_agent_config()
        self.mock_logger.error.assert_called_with(
            'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
        )

    def test_load_config_from_file(self):
        config_yaml = """
        api_key: test-key
        api_secret: test-secret
        ws_url: test-url
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = open(os.path.join(temp_dir, "agent-MY_AGENT_NAME.yml"), "w")
            temp_file.write(config_yaml)
            temp_file.close()
            os.environ["AGENT_CONFIG_FILE"] = temp_file.name
            agent_config, agent_name = load_agent_config()

            self.assertEqual(agent_name, "MY_AGENT_NAME")
            self.assertEqual(agent_config["api_key"], "test-key")
            self.assertEqual(agent_config["api_secret"], "test-secret")
            self.assertEqual(agent_config["ws_url"], "test-url")

    def test_load_config_from_file_with_wrong_agent_name(self):
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: test-url
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = open(os.path.join(temp_dir, "agent-MY_AGENT_NAME.yml"), "w")
            temp_file.write(config_yaml)
            temp_file.close()
            os.environ["AGENT_CONFIG_FILE"] = temp_file.name
            with self.assertRaises(SystemExit):
                load_agent_config()
            self.mock_logger.error.assert_called_with(
                f'Agent name is defined as "test-agent" inside configuration file {temp_file.name} and it does not match the value of the file name "MY_AGENT_NAME"'
            )

    def test_load_config_from_file_with_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yml") as temp_file:
            temp_file.write(b"invalid: yaml: ::")
            temp_file.close()
            os.environ["AGENT_CONFIG_FILE"] = temp_file.name
            with self.assertRaises(SystemExit):
                load_agent_config()
            self.mock_logger.error.assert_called_with(
                f"Env var AGENT_CONFIG_FILE set to {temp_file.name}, but file is not a valid agent configuration file. It must exist and be named agent-AGENT_NAME.yml"
            )
        os.unlink(temp_file.name)

    def test_is_agent_config_file(self):
        valid_files = ["agent-test.yml", "agent-test.yaml"]
        invalid_files = ["invalid.yml", "agent_test.yml", "agent-.yml"]

        for file in valid_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file)
                with open(file_path, "w") as f:
                    f.write("test: value")
                self.assertTrue(is_agent_config_file(temp_dir, file))

        for file in invalid_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file)
                with open(file_path, "w") as f:
                    f.write("test: value")
                self.assertFalse(is_agent_config_file(temp_dir, file))


if __name__ == "__main__":
    unittest.main()
