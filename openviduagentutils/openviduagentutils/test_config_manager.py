import unittest
from enum import Enum
import copy
from config_manager import ConfigManager

# Mock the __main__ module to avoid import issues
# sys.modules["__main__"] = sys.modules[__name__]

# Import the ConfigManager class (assuming it's in a file named config_manager.py)
# If it's in a different file, adjust the import statement accordingly
from typing import TypeVar
from enum import Enum

T = TypeVar("T", bound=Enum)


class TestConfigManager(unittest.TestCase):
    """
    Test class for the ConfigManager class.
    Tests all methods and edge cases.
    """

    def setUp(self):
        """Set up test data before each test."""
        # Basic configuration for testing
        self.basic_config = {
            "server": {
                "host": "localhost",
                "port": 8080,
                "debug": True,
                "null_value": None,
            },
            "database": {
                "url": "postgres://user:password@localhost:5432/db",
                "max_connections": 10,
                "settings": {
                    "timeout": 30,
                    "retry": 3,
                },
            },
            "logging": {
                "level": "INFO",
                "file": "/var/log/app.log",
                "enabled": True,
            },
            "empty_dict": {},
            "list_value": [1, 2, 3],
            "string_value": "test",
            "int_value": 42,
            "bool_value": True,
            "null_value": None,
        }

        # Define a test enum
        class LogLevel(Enum):
            DEBUG = "DEBUG"
            INFO = "INFO"
            WARNING = "WARNING"
            ERROR = "ERROR"
            CRITICAL = "CRITICAL"

        self.LogLevel = LogLevel

    def test_initialization_valid_cases(self):
        """Test initialization with valid parameters."""
        # Test with valid config and string prefix
        config_manager = ConfigManager(self.basic_config, "prefix")
        self.assertEqual(config_manager.config, self.basic_config)
        self.assertEqual(config_manager.field_prefixes, "prefix")

        # Test with valid config and empty string prefix
        config_manager = ConfigManager(self.basic_config, "")
        self.assertEqual(config_manager.config, self.basic_config)
        self.assertEqual(config_manager.field_prefixes, "")

        # Test with valid config and None prefix
        config_manager = ConfigManager(self.basic_config, None)
        self.assertEqual(config_manager.config, self.basic_config)
        self.assertEqual(config_manager.field_prefixes, "")

        # Test with empty config
        config_manager = ConfigManager({}, "prefix")
        self.assertEqual(config_manager.config, {})
        self.assertEqual(config_manager.field_prefixes, "prefix")

    def test_initialization_invalid_cases(self):
        """Test initialization with invalid parameters."""
        # Test with non-dictionary config
        with self.assertRaises(TypeError):
            ConfigManager("not a dict", "prefix")

        with self.assertRaises(TypeError):
            ConfigManager(123, "prefix")

        with self.assertRaises(TypeError):
            ConfigManager(None, "prefix")

        # Test with non-string prefix
        with self.assertRaises(TypeError):
            ConfigManager({}, 123)

        with self.assertRaises(TypeError):
            ConfigManager({}, ["prefix"])

    def test_full_key(self):
        """Test the __full_key method."""
        # Test with prefix
        config_manager = ConfigManager({}, "prefix")
        self.assertEqual(config_manager._ConfigManager__full_key("key"), "prefix.key")

        # Test with empty prefix
        config_manager = ConfigManager({}, "")
        self.assertEqual(config_manager._ConfigManager__full_key("key"), "key")

        # Test with None prefix (should be converted to empty string)
        config_manager = ConfigManager({}, None)
        self.assertEqual(config_manager._ConfigManager__full_key("key"), "key")

    def test_get_value_valid_cases(self):
        """Test the __get_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test simple key
        self.assertEqual(
            config_manager._ConfigManager__get_value("string_value"), "test"
        )
        self.assertEqual(config_manager._ConfigManager__get_value("int_value"), 42)
        self.assertEqual(config_manager._ConfigManager__get_value("bool_value"), True)
        self.assertIsNone(config_manager._ConfigManager__get_value("null_value"))

        # Test nested keys
        self.assertEqual(
            config_manager._ConfigManager__get_value("server.host"), "localhost"
        )
        self.assertEqual(config_manager._ConfigManager__get_value("server.port"), 8080)
        self.assertEqual(
            config_manager._ConfigManager__get_value("database.settings.timeout"), 30
        )
        self.assertEqual(
            config_manager._ConfigManager__get_value("database.settings.retry"), 3
        )

        # Test accessing dict values
        self.assertEqual(
            config_manager._ConfigManager__get_value("database.settings"),
            {"timeout": 30, "retry": 3},
        )
        self.assertEqual(
            config_manager._ConfigManager__get_value("server"),
            {
                "host": "localhost",
                "port": 8080,
                "debug": True,
                "null_value": None,
            },
        )

        # Test accessing empty dict
        self.assertEqual(config_manager._ConfigManager__get_value("empty_dict"), {})

        # Test accessing list
        self.assertEqual(
            config_manager._ConfigManager__get_value("list_value"), [1, 2, 3]
        )

    def test_get_value_invalid_cases(self):
        """Test the __get_value method with invalid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test non-existent key
        with self.assertRaises(KeyError):
            config_manager._ConfigManager__get_value("non_existent")

        # Test non-existent nested key
        with self.assertRaises(KeyError):
            config_manager._ConfigManager__get_value("server.non_existent")

        # Test accessing a value that is not a dictionary as if it were
        with self.assertRaises(TypeError):
            config_manager._ConfigManager__get_value("string_value.invalid")

        # Test accessing a list index (should fail because lists are not supported)
        with self.assertRaises(TypeError):
            config_manager._ConfigManager__get_value("list_value.0")

    def test_mandatory_value_valid_cases(self):
        """Test the mandatory_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test simple key
        self.assertEqual(
            config_manager.mandatory_value("string_value", "Error"), "test"
        )
        self.assertEqual(config_manager.mandatory_value("int_value", "Error"), 42)
        self.assertEqual(config_manager.mandatory_value("bool_value", "Error"), True)

        # Test nested keys
        self.assertEqual(
            config_manager.mandatory_value("server.host", "Error"), "localhost"
        )
        self.assertEqual(config_manager.mandatory_value("server.port", "Error"), 8080)
        self.assertEqual(
            config_manager.mandatory_value("database.settings.timeout", "Error"), 30
        )

        # Test with prefix
        config_manager = ConfigManager(self.basic_config, "prefix")
        self.assertEqual(
            config_manager._ConfigManager__full_key("string_value"),
            "prefix.string_value",
        )

    def test_mandatory_value_invalid_cases(self):
        """Test the mandatory_value method with invalid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test non-existent key
        with self.assertRaises(ValueError) as context:
            config_manager.mandatory_value("non_existent", "Custom error message")
        self.assertEqual(str(context.exception), "Custom error message")

        # Test non-existent nested key
        with self.assertRaises(ValueError) as context:
            config_manager.mandatory_value(
                "server.non_existent", "Custom error message"
            )
        self.assertEqual(str(context.exception), "Custom error message")

        # Test accessing a value that is not a dictionary as if it were
        with self.assertRaises(ValueError) as context:
            config_manager.mandatory_value(
                "string_value.invalid", "Custom error message"
            )
        self.assertEqual(str(context.exception), "Custom error message")

        # Test null value
        with self.assertRaises(ValueError) as context:
            config_manager.mandatory_value("null_value", "Custom error message")
        self.assertEqual(str(context.exception), "Custom error message")

        # Test nested null value
        with self.assertRaises(ValueError) as context:
            config_manager.mandatory_value("server.null_value", "Custom error message")
        self.assertEqual(str(context.exception), "Custom error message")

    def test_optional_value_valid_cases(self):
        """Test the optional_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test simple key
        self.assertEqual(
            config_manager.optional_value("string_value", "default"), "test"
        )
        self.assertEqual(config_manager.optional_value("int_value", 0), 42)
        self.assertEqual(config_manager.optional_value("bool_value", False), True)

        # Test nested keys
        self.assertEqual(
            config_manager.optional_value("server.host", "default"), "localhost"
        )
        self.assertEqual(config_manager.optional_value("server.port", 0), 8080)
        self.assertEqual(
            config_manager.optional_value("database.settings.timeout", 0), 30
        )

        # Test with null values (should return default)
        self.assertEqual(
            config_manager.optional_value("null_value", "default"), "default"
        )
        self.assertEqual(
            config_manager.optional_value("server.null_value", "default"), "default"
        )

        # Test with non-existent keys (should return default)
        self.assertEqual(
            config_manager.optional_value("non_existent", "default"), "default"
        )
        self.assertEqual(
            config_manager.optional_value("server.non_existent", "default"), "default"
        )

        # Test with non-dict parent (should return default)
        self.assertEqual(
            config_manager.optional_value("string_value.invalid", "default"), "default"
        )

    def test_optional_string_value_valid_cases(self):
        """Test the optional_string_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with string value
        self.assertEqual(config_manager.optional_string_value("string_value"), "test")
        self.assertEqual(
            config_manager.optional_string_value("string_value", "default"), "test"
        )

        # Test with nested string value
        self.assertEqual(
            config_manager.optional_string_value("server.host"), "localhost"
        )
        self.assertEqual(
            config_manager.optional_string_value("server.host", "default"), "localhost"
        )

        # Test with null value (should return default)
        self.assertIsNone(config_manager.optional_string_value("null_value"))
        self.assertEqual(
            config_manager.optional_string_value("null_value", "default"), "default"
        )

        # Test with non-existent key (should return default)
        self.assertIsNone(config_manager.optional_string_value("non_existent"))
        self.assertEqual(
            config_manager.optional_string_value("non_existent", "default"), "default"
        )

    def test_optional_string_value_invalid_cases(self):
        """Test the optional_string_value method with invalid paths or types."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with non-string value
        with self.assertRaises(TypeError) as context:
            config_manager.optional_string_value("int_value")
        self.assertIn("must be a string", str(context.exception))

        # Test with non-string nested value
        with self.assertRaises(TypeError) as context:
            config_manager.optional_string_value("server.port")
        self.assertIn("must be a string", str(context.exception))

        # Test with non-dict parent (should return default without error)
        self.assertEqual(
            config_manager.optional_string_value("string_value.invalid", "default"),
            "default",
        )

    def test_optional_enum_value_valid_cases(self):
        """Test the optional_enum_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with valid enum value
        self.assertEqual(
            config_manager.optional_enum_value("logging.level", self.LogLevel),
            self.LogLevel.INFO,
        )
        self.assertEqual(
            config_manager.optional_enum_value(
                "logging.level", self.LogLevel, self.LogLevel.DEBUG
            ),
            self.LogLevel.INFO,
        )

        # Test with null value (should return default)
        self.assertIsNone(
            config_manager.optional_enum_value("null_value", self.LogLevel)
        )
        self.assertEqual(
            config_manager.optional_enum_value(
                "null_value", self.LogLevel, self.LogLevel.ERROR
            ),
            self.LogLevel.ERROR,
        )

        # Test with non-existent key (should return default)
        self.assertIsNone(
            config_manager.optional_enum_value("non_existent", self.LogLevel)
        )
        self.assertEqual(
            config_manager.optional_enum_value(
                "non_existent", self.LogLevel, self.LogLevel.ERROR
            ),
            self.LogLevel.ERROR,
        )

    def test_optional_enum_value_invalid_cases(self):
        """Test the optional_enum_value method with invalid paths or types."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with invalid enum value
        with self.assertRaises(ValueError) as context:
            config_manager.optional_enum_value("string_value", self.LogLevel)
        self.assertIn("Invalid value", str(context.exception))
        self.assertIn("Valid values are", str(context.exception))

        # Test with non-string value
        with self.assertRaises(ValueError) as context:
            config_manager.optional_enum_value("int_value", self.LogLevel)
        self.assertIn("Invalid value", str(context.exception))

        # Test with non-dict parent (should return default without error)
        self.assertEqual(
            config_manager.optional_enum_value(
                "string_value.invalid", self.LogLevel, self.LogLevel.ERROR
            ),
            self.LogLevel.ERROR,
        )

    def test_optional_dict_value_valid_cases(self):
        """Test the optional_dict_value method with valid paths."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with dict value
        expected_dict = {"timeout": 30, "retry": 3}
        self.assertEqual(
            config_manager.optional_dict_value("database.settings"), expected_dict
        )
        self.assertEqual(
            config_manager.optional_dict_value("database.settings", {"default": True}),
            expected_dict,
        )

        # Test with empty dict
        self.assertEqual(config_manager.optional_dict_value("empty_dict"), {})
        self.assertEqual(
            config_manager.optional_dict_value("empty_dict", {"default": True}), {}
        )

        # Test with null value (should return default or empty dict)
        self.assertEqual(config_manager.optional_dict_value("null_value"), {})
        self.assertEqual(
            config_manager.optional_dict_value("null_value", {"default": True}),
            {"default": True},
        )

        # Test with non-existent key (should return default or empty dict)
        self.assertEqual(config_manager.optional_dict_value("non_existent"), {})
        self.assertEqual(
            config_manager.optional_dict_value("non_existent", {"default": True}),
            {"default": True},
        )

        # Test with explicitly passed None as default (should return empty dict)
        self.assertEqual(config_manager.optional_dict_value("non_existent", None), {})

    def test_optional_dict_value_invalid_cases(self):
        """Test the optional_dict_value method with invalid paths or types."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test with non-dict value
        with self.assertRaises(TypeError) as context:
            config_manager.optional_dict_value("string_value")
        self.assertIn("must be a dictionary", str(context.exception))

        # Test with non-dict nested value
        with self.assertRaises(TypeError) as context:
            config_manager.optional_dict_value("server.port")
        self.assertIn("must be a dictionary", str(context.exception))

        # Test with non-dict parent (should return default without error)
        self.assertEqual(config_manager.optional_dict_value("string_value.invalid"), {})
        self.assertEqual(
            config_manager.optional_dict_value(
                "string_value.invalid", {"default": True}
            ),
            {"default": True},
        )

    def test_mandatory_value_with_modified_config(self):
        """Test that mandatory_value properly reflects changes to the config."""
        config_manager = ConfigManager(self.basic_config, "")

        # Modify the config after creating the manager
        modified_config = copy.deepcopy(self.basic_config)
        modified_config["string_value"] = "modified"
        config_manager.config = modified_config

        # Test that the change is reflected
        self.assertEqual(
            config_manager.mandatory_value("string_value", "Error"), "modified"
        )

    def test_with_deeper_nested_structures(self):
        """Test with deeper nested structures."""
        deep_config = {
            "level1": {
                "level2": {"level3": {"level4": {"value": "deep", "number": 42}}}
            }
        }

        config_manager = ConfigManager(deep_config, "")

        # Test deep mandatory value
        self.assertEqual(
            config_manager.mandatory_value(
                "level1.level2.level3.level4.value", "Error"
            ),
            "deep",
        )

        # Test deep optional value
        self.assertEqual(
            config_manager.optional_value("level1.level2.level3.level4.number", 0), 42
        )

        # Test deep optional string value
        self.assertEqual(
            config_manager.optional_string_value("level1.level2.level3.level4.value"),
            "deep",
        )

        # Test deep optional dict value
        expected = {"value": "deep", "number": 42}
        self.assertEqual(
            config_manager.optional_dict_value("level1.level2.level3.level4"), expected
        )

    def test_with_empty_config(self):
        """Test with an empty configuration."""
        config_manager = ConfigManager({}, "")

        # Test mandatory value with empty config
        with self.assertRaises(ValueError):
            config_manager.mandatory_value("key", "Error")

        # Test optional value with empty config
        self.assertEqual(config_manager.optional_value("key", "default"), "default")

        # Test optional string value with empty config
        self.assertEqual(
            config_manager.optional_string_value("key", "default"), "default"
        )

        # Test optional enum value with empty config
        self.assertEqual(
            config_manager.optional_enum_value(
                "key", self.LogLevel, self.LogLevel.ERROR
            ),
            self.LogLevel.ERROR,
        )

        # Test optional dict value with empty config
        self.assertEqual(config_manager.optional_dict_value("key"), {})
        self.assertEqual(
            config_manager.optional_dict_value("key", {"default": True}),
            {"default": True},
        )

    def test_with_complex_default_values(self):
        """Test with complex default values."""
        config_manager = ConfigManager(self.basic_config, "")

        # Test optional value with complex default
        complex_default = {"nested": {"value": [1, 2, 3]}}
        self.assertEqual(
            config_manager.optional_value("non_existent", complex_default),
            complex_default,
        )

        # Test that the default is returned by reference, not by value
        result = config_manager.optional_value("non_existent", complex_default)
        self.assertIs(result, complex_default)

        # Test optional dict value with complex default
        complex_dict_default = {"nested": {"value": [1, 2, 3]}}
        self.assertEqual(
            config_manager.optional_dict_value("non_existent", complex_dict_default),
            complex_dict_default,
        )

        # Test that the default is returned by reference, not by value
        result = config_manager.optional_dict_value(
            "non_existent", complex_dict_default
        )
        self.assertIs(result, complex_dict_default)

    def test_with_keys_containing_dots(self):
        """Test with keys containing dots."""
        config_with_dots = {"server": {"host.name": "localhost"}}

        config_manager = ConfigManager(config_with_dots, "")

        # This should fail since keys cannot have dots
        with self.assertRaises(ValueError):
            config_manager.mandatory_value("server.host.name", "Error")


if __name__ == "__main__":
    unittest.main()
