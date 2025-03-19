from io import StringIO
import unittest
import os
import tempfile
import logging
from unittest.mock import patch, mock_open

from config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.config_loader = ConfigLoader()
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.DEBUG)

        # Clean environment variables before each test
        env_vars = [
            "AGENT_CONFIG_BODY",
            "AGENT_CONFIG_FILE",
            "ENV_VARS_FILE",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "LIVEKIT_URL",
            "REDIS_ADDRESS",
            "REDIS_DB",
            "REDIS_USERNAME",
            "REDIS_PASSWORD",
            "REDIS_SENTINEL_MASTER_NAME",
            "REDIS_SENTINEL_ADDRESSES",
            "REDIS_SENTINEL_USERNAME",
            "REDIS_SENTINEL_PASSWORD",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        # Clean up logging configuration
        logging.getLogger().removeHandler(self.log_handler)
        logging.getLogger().setLevel(logging.NOTSET)

    def test_load_from_env_var_body(self):
        """Test loading configuration from AGENT_CONFIG_BODY environment variable."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(name, "test-agent")
            self.assertEqual(config["api_key"], "test-key")
            self.assertEqual(config["api_secret"], "test-secret")
            self.assertEqual(config["ws_url"], "ws://test.url")
            self.assertEqual(config["redis"]["address"], "localhost:6379")
            self.assertEqual(config["redis"]["db"], 0)
            self.assertEqual(config["redis"]["password"], "pass")

    def test_load_from_env_var_file(self):
        """Test loading configuration from AGENT_CONFIG_FILE environment variable."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
            temp_file.write(config_yaml.encode())
            temp_file_path = temp_file.name

        try:
            # Create a temporary file with the format agent-NAME.yml
            temp_dir = os.path.dirname(temp_file_path)
            agent_file = os.path.join(temp_dir, "agent-test-agent.yml")
            with open(agent_file, "w") as f:
                f.write(config_yaml)

            with patch.dict(os.environ, {"AGENT_CONFIG_FILE": agent_file}):
                config, name = self.config_loader.load_agent_config()

                self.assertEqual(name, "test-agent")
                self.assertEqual(config["api_key"], "test-key")
                self.assertEqual(config["api_secret"], "test-secret")
                self.assertEqual(config["ws_url"], "ws://test.url")
                self.assertEqual(config["redis"]["address"], "localhost:6379")
                self.assertEqual(config["redis"]["db"], 0)
                self.assertEqual(config["redis"]["password"], "pass")
        finally:
            # Clean up the temporary files
            os.unlink(temp_file_path)
            if os.path.exists(agent_file):
                os.unlink(agent_file)

    @patch("os.getcwd")
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_load_from_cwd(self, mock_isfile, mock_listdir, mock_getcwd):
        """Test loading configuration from current working directory."""
        mock_getcwd.return_value = "/fake/path"
        mock_listdir.return_value = ["agent-test-agent.yml", "something-else.txt"]
        mock_isfile.return_value = True

        config_yaml = """
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with patch("builtins.open", mock_open(read_data=config_yaml)):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(name, "test-agent")
            self.assertEqual(config["api_key"], "test-key")
            self.assertEqual(config["api_secret"], "test-secret")
            self.assertEqual(config["ws_url"], "ws://test.url")

    @patch("os.getcwd")
    @patch("os.listdir")
    @patch("os.path.isfile")
    @patch("sys.argv", ["/path/to/script.py"])
    def test_load_from_script_dir(self, mock_isfile, mock_listdir, mock_getcwd):
        """Test loading configuration from script directory when not in cwd."""
        mock_getcwd.return_value = "/fake/path"
        # First listdir (cwd) returns no config files
        # Second listdir (script dir) returns config file
        mock_listdir.side_effect = [["something-else.txt"], ["agent-test-agent.yml"]]
        mock_isfile.return_value = True

        config_yaml = """
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with patch("builtins.open", mock_open(read_data=config_yaml)):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(name, "test-agent")
            self.assertEqual(config["api_key"], "test-key")
            self.assertEqual(config["api_secret"], "test-secret")
            self.assertEqual(config["ws_url"], "ws://test.url")

    def test_invalid_yaml_in_env_body(self):
        """Test handling invalid YAML in AGENT_CONFIG_BODY."""
        invalid_yaml = "invalid: yaml: structure:"

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": invalid_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_missing_agent_name_in_env_body(self):
        """Test handling missing agent_name in AGENT_CONFIG_BODY."""
        config_yaml = """
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_null_agent_name_in_env_body(self):
        """Test handling null agent_name in AGENT_CONFIG_BODY."""
        config_yaml = """
        agent_name: null
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    @patch("os.path.isfile")
    def test_invalid_agent_config_file(self, mock_isfile):
        """Test handling invalid AGENT_CONFIG_FILE."""
        mock_isfile.return_value = True

        with patch.dict(
            os.environ, {"AGENT_CONFIG_FILE": "/path/to/not-agent-file.yml"}
        ):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_invalid_agent_config_from_file(self):
        """Test handling of invalid YAML in config file."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
            temp_file.write(b"invalid: yaml: structure:")
            temp_file_path = temp_file.name

        try:
            # Create a temporary file with the format agent-NAME.yml
            temp_dir = os.path.dirname(temp_file_path)
            agent_file = os.path.join(temp_dir, "agent-test-agent.yml")
            with open(agent_file, "w") as f:
                f.write("invalid: yaml: structure:")

            with patch.dict(os.environ, {"AGENT_CONFIG_FILE": agent_file}):
                with self.assertRaises(SystemExit):
                    self.config_loader.load_agent_config()
        finally:
            # Clean up the temporary files
            os.unlink(temp_file_path)
            if os.path.exists(agent_file):
                os.unlink(agent_file)

    def test_agent_name_mismatch(self):
        """Test handling of agent_name mismatch between file name and config."""
        config_yaml = """
        agent_name: mismatched-name
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
            temp_file.write(config_yaml.encode())
            temp_file_path = temp_file.name

        try:
            # Create a temporary file with the format agent-NAME.yml
            temp_dir = os.path.dirname(temp_file_path)
            agent_file = os.path.join(temp_dir, "agent-test-agent.yml")
            with open(agent_file, "w") as f:
                f.write(config_yaml)

            with patch.dict(os.environ, {"AGENT_CONFIG_FILE": agent_file}):
                with self.assertRaises(SystemExit):
                    self.config_loader.load_agent_config()
        finally:
            # Clean up the temporary files
            os.unlink(temp_file_path)
            if os.path.exists(agent_file):
                os.unlink(agent_file)

    def test_fallback_to_env_vars_for_connection(self):
        """Test fallback to environment variables for connection details."""
        config_yaml = """
        agent_name: test-agent
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with patch.dict(
            os.environ,
            {
                "AGENT_CONFIG_BODY": config_yaml,
                "LIVEKIT_API_KEY": "env-api-key",
                "LIVEKIT_API_SECRET": "env-api-secret",
                "LIVEKIT_URL": "env-ws-url",
            },
        ):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(config["api_key"], "env-api-key")
            self.assertEqual(config["api_secret"], "env-api-secret")
            self.assertEqual(config["ws_url"], "env-ws-url")

    def test_missing_connection_details(self):
        """Test handling of missing connection details."""
        config_yaml = """
        agent_name: test-agent
        redis:
            address: localhost:6379
            db: 0
            password: pass
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_redis_config_from_env_standalone(self):
        """Test loading Redis standalone configuration from environment variables."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with patch.dict(
            os.environ,
            {
                "AGENT_CONFIG_BODY": config_yaml,
                "REDIS_ADDRESS": "localhost:6379",
                "REDIS_PASSWORD": "pass",
                "REDIS_USERNAME": "user",
                "REDIS_DB": "1",
            },
        ):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(config["redis"]["address"], "localhost:6379")
            self.assertEqual(config["redis"]["password"], "pass")
            self.assertEqual(config["redis"]["username"], "user")
            self.assertEqual(config["redis"]["db"], "1")

    def test_redis_config_from_env_sentinel(self):
        """Test loading Redis sentinel configuration from environment variables."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with patch.dict(
            os.environ,
            {
                "AGENT_CONFIG_BODY": config_yaml,
                "REDIS_SENTINEL_ADDRESSES": "sentinel1:26379,sentinel2:26379",
                "REDIS_SENTINEL_MASTER_NAME": "master",
                "REDIS_SENTINEL_PASSWORD": "pass",
                "REDIS_SENTINEL_USERNAME": "user",
                "REDIS_DB": "1",
            },
        ):
            config, name = self.config_loader.load_agent_config()

            self.assertEqual(
                config["redis"]["sentinel_addresses"],
                ["sentinel1:26379", "sentinel2:26379"],
            )
            self.assertEqual(config["redis"]["sentinel_master_name"], "master")
            self.assertEqual(config["redis"]["sentinel_password"], "pass")
            self.assertEqual(config["redis"]["sentinel_username"], "user")
            self.assertEqual(config["redis"]["db"], "1")

    def test_invalid_redis_config_missing_db(self):
        """Test handling of invalid Redis configuration with missing db."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            address: localhost:6379
            password: pass
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_invalid_redis_config_missing_connection(self):
        """Test handling of invalid Redis configuration with missing connection details."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            db: 0
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_invalid_redis_config_missing_sentinel_details(self):
        """Test handling of invalid Redis sentinel configuration with missing details."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            db: 0
            sentinel_addresses:
                - sentinel1:26379
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_invalid_redis_config_missing_address_password(self):
        """Test handling of invalid Redis configuration with missing password."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        redis:
            db: 0
            address: localhost:6379
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(SystemExit):
                self.config_loader.load_agent_config()

    def test_missing_redis_env_vars(self):
        """Test handling of missing Redis environment variables."""
        config_yaml = """
        agent_name: test-agent
        api_key: test-key
        api_secret: test-secret
        ws_url: ws://test.url
        """

        with patch.dict(os.environ, {"AGENT_CONFIG_BODY": config_yaml}):
            with self.assertRaises(ValueError):
                self.config_loader.load_agent_config()

    def test_is_agent_config_file(self):
        """Test the __is_agent_config_file private method."""
        # Use a different method to test the private method
        with patch("os.path.isfile") as mock_isfile:
            mock_isfile.return_value = True

            # Test valid file names
            self.assertTrue(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test.yml"
                )
            )
            self.assertTrue(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test.yaml"
                )
            )
            self.assertTrue(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test-with-dashes.yml"
                )
            )
            self.assertTrue(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test_with_underscores.yml"
                )
            )

            # Test invalid file names
            self.assertFalse(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "not-agent-test.yml"
                )
            )
            self.assertFalse(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test.txt"
                )
            )

            # Test non-existent file
            mock_isfile.return_value = False
            self.assertFalse(
                self.config_loader._ConfigLoader__is_agent_config_file(
                    "/fake/path", "agent-test.yml"
                )
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_env_vars_file_not_defined(self):
        """Test when ENV_VARS_FILE environment variable is not defined."""
        self.config_loader._ConfigLoader__load_env_vars_from_file()
        self.assertIn("ENV_VARS_FILE not defined", self.log_stream.getvalue())

    @patch.dict(os.environ, {"ENV_VARS_FILE": "/path/to/nonexistent/file"})
    @patch("os.path.isfile", return_value=False)
    def test_env_vars_file_not_exists(self, mock_isfile):
        """Test when ENV_VARS_FILE points to a non-existent file."""
        self.config_loader._ConfigLoader__load_env_vars_from_file()
        self.assertIn(
            "ENV_VARS_FILE defined but is not a file", self.log_stream.getvalue()
        )

    @patch.dict(os.environ, {"ENV_VARS_FILE": "/path/to/unreadable/file"})
    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=False)
    def test_env_vars_file_not_readable(self, mock_access, mock_isfile):
        """Test when ENV_VARS_FILE points to a file that is not readable."""
        self.config_loader._ConfigLoader__load_env_vars_from_file()
        self.assertIn(
            "ENV_VARS_FILE defined but is not readable", self.log_stream.getvalue()
        )

    @patch.dict(os.environ, {"ENV_VARS_FILE": "/path/to/valid/file"})
    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    @patch("config_loader.load_dotenv", return_value=True)
    def test_env_vars_file_loads_successfully(
        self, mock_load_dotenv, mock_access, mock_isfile
    ):
        """Test when ENV_VARS_FILE points to a valid file that loads successfully."""
        self.config_loader._ConfigLoader__load_env_vars_from_file()
        self.assertIn(
            "Loading environment variables from file", self.log_stream.getvalue()
        )
        mock_load_dotenv.assert_called_once_with("/path/to/valid/file")

    @patch.dict(os.environ, {"ENV_VARS_FILE": "/path/to/empty/file"})
    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    @patch("config_loader.load_dotenv", return_value=False)
    def test_env_vars_file_empty(self, mock_load_dotenv, mock_access, mock_isfile):
        """Test when ENV_VARS_FILE points to a file with no environment variables."""
        self.config_loader._ConfigLoader__load_env_vars_from_file()
        self.assertIn("No env vars available at file", self.log_stream.getvalue())
        mock_load_dotenv.assert_called_once_with("/path/to/empty/file")


if __name__ == "__main__":
    unittest.main()
