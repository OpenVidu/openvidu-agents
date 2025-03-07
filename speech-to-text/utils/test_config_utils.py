import unittest
from enum import Enum
from typing import Dict, Any

# Import the ConfigManager class - assuming it's in a module called config_manager
from config_utils import ConfigManager


class TestEnum(Enum):
    """Enum for testing optional_enum_value method."""

    OPTION1 = 1
    OPTION2 = 2
    OPTION3 = 3


class TestConfigManager(unittest.TestCase):
    """Test suite for the ConfigManager class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Sample configuration for testing
        self.config: Dict[str, Any] = {
            "section1": {
                "key1": "value1",
                "key2": None,
                "subsection": {"key3": "value3", "nested": {"key4": "value4"}},
                "list_value": [1, 2, 3],
                "bool_value": True,
                "enum_value": "OPTION1",
                "invalid_enum": "INVALID",
            },
            "section2": {"key1": 123, "key2": False, "not_dict": "string"},
            "empty_section": {},
        }
        self.config_manager = ConfigManager(self.config, "section1")
        self.config_manager_no_prefix = ConfigManager(self.config, "")
        self.config_manager_none_prefix = ConfigManager(self.config, None)

    def test_init_valid(self) -> None:
        """Test initialization with valid parameters."""
        cm = ConfigManager(self.config, "section1")
        self.assertEqual(cm.config, self.config)
        self.assertEqual(cm.field_prefixes, "section1")

        # Test with empty prefix
        cm = ConfigManager(self.config, "")
        self.assertEqual(cm.field_prefixes, "")

        # Test with None prefix
        cm = ConfigManager(self.config, None)
        self.assertEqual(cm.field_prefixes, "")

    def test_init_invalid_config(self) -> None:
        """Test initialization with invalid config parameter."""
        with self.assertRaises(TypeError) as context:
            ConfigManager("not_a_dict", "section1")
        self.assertEqual(str(context.exception), "config must be a dictionary")

        with self.assertRaises(TypeError) as context:
            ConfigManager(123, "section1")
        self.assertEqual(str(context.exception), "config must be a dictionary")

        with self.assertRaises(TypeError) as context:
            ConfigManager(None, "section1")
        self.assertEqual(str(context.exception), "config must be a dictionary")

    def test_init_invalid_prefix(self) -> None:
        """Test initialization with invalid field_prefixes parameter."""
        with self.assertRaises(TypeError) as context:
            ConfigManager({}, 123)
        self.assertEqual(
            str(context.exception), "field_prefixes must be a string or None"
        )

        with self.assertRaises(TypeError) as context:
            ConfigManager({}, {})
        self.assertEqual(
            str(context.exception), "field_prefixes must be a string or None"
        )

    def test_mandatory_value_success(self) -> None:
        """Test successful retrieval of mandatory values."""
        # Direct key
        value = self.config_manager.mandatory_value("key1", "Error")
        self.assertEqual(value, "value1")

        # Nested key
        value = self.config_manager.mandatory_value("subsection.key3", "Error")
        self.assertEqual(value, "value3")

        # Deeply nested key
        value = self.config_manager.mandatory_value("subsection.nested.key4", "Error")
        self.assertEqual(value, "value4")

        # None value (should not raise error)
        value = self.config_manager.mandatory_value("key2", "Error")
        self.assertIsNone(value)

        # Non-string values
        value = self.config_manager_no_prefix.mandatory_value("section2.key1", "Error")
        self.assertEqual(value, 123)

        value = self.config_manager_no_prefix.mandatory_value("section2.key2", "Error")
        self.assertEqual(value, False)

    def test_mandatory_value_failure(self) -> None:
        """Test error cases for mandatory_value method."""
        # Non-existent key
        with self.assertRaises(ValueError) as context:
            self.config_manager.mandatory_value("nonexistent", "Custom error")
        self.assertEqual(str(context.exception), "Custom error")

        # Non-existent nested key
        with self.assertRaises(ValueError) as context:
            self.config_manager.mandatory_value(
                "subsection.nonexistent", "Custom error"
            )
        self.assertEqual(str(context.exception), "Custom error")

        # Key in non-dictionary
        with self.assertRaises(ValueError) as context:
            self.config_manager_no_prefix.mandatory_value(
                "section2.not_dict.something", "Custom error"
            )
        self.assertEqual(str(context.exception), "Custom error")

    def test_optional_value_success(self) -> None:
        """Test successful retrieval of optional values."""
        # Direct key
        value = self.config_manager.optional_value("key1", "default")
        self.assertEqual(value, "value1")

        # Nested key
        value = self.config_manager.optional_value("subsection.key3", "default")
        self.assertEqual(value, "value3")

        # None value (should return default)
        value = self.config_manager.optional_value("key2", "default")
        self.assertEqual(value, "default")

        # Empty string default
        value = self.config_manager.optional_value("nonexistent", "")
        self.assertEqual(value, "")

        # None default
        value = self.config_manager.optional_value("nonexistent", None)
        self.assertIsNone(value)

        # Zero default
        value = self.config_manager.optional_value("nonexistent", 0)
        self.assertEqual(value, 0)

        # False default
        value = self.config_manager.optional_value("nonexistent", False)
        self.assertEqual(value, False)

    def test_optional_value_with_different_types(self) -> None:
        """Test optional_value with different value types."""
        # List value
        value = self.config_manager.optional_value("list_value", None)
        self.assertEqual(value, [1, 2, 3])

        # Boolean value
        value = self.config_manager.optional_value("bool_value", None)
        self.assertTrue(value)

        # Integer value
        value = self.config_manager_no_prefix.optional_value("section2.key1", None)
        self.assertEqual(value, 123)

    def test_optional_value_failure_cases(self) -> None:
        """Test error handling in optional_value method."""
        # Non-existent key
        value = self.config_manager.optional_value("nonexistent", "default")
        self.assertEqual(value, "default")

        # Non-existent nested key
        value = self.config_manager.optional_value("subsection.nonexistent", "default")
        self.assertEqual(value, "default")

        # Key in non-dictionary (should return default)
        value = self.config_manager_no_prefix.optional_value(
            "section2.not_dict.something", "default"
        )
        self.assertEqual(value, "default")

    def test_optional_enum_value_success(self) -> None:
        """Test successful retrieval of enum values."""
        # Valid enum value
        value = self.config_manager.optional_enum_value("enum_value", TestEnum)
        self.assertEqual(value, TestEnum.OPTION1)

        # With default
        value = self.config_manager.optional_enum_value(
            "nonexistent", TestEnum, TestEnum.OPTION2
        )
        self.assertEqual(value, TestEnum.OPTION2)

        # With None default
        value = self.config_manager.optional_enum_value("nonexistent", TestEnum, None)
        self.assertIsNone(value)

        # Non-existent key - default to None
        value = self.config_manager.optional_enum_value("nonexistent", TestEnum)
        self.assertIsNone(value)

    def test_optional_enum_value_invalid(self) -> None:
        """Test error handling for invalid enum values."""
        # Invalid enum value
        with self.assertRaises(ValueError) as context:
            self.config_manager.optional_enum_value("invalid_enum", TestEnum)

        error_msg = str(context.exception)
        self.assertIn("Invalid value 'INVALID'", error_msg)
        self.assertIn("Valid values are:", error_msg)
        self.assertIn("OPTION1", error_msg)
        self.assertIn("OPTION2", error_msg)
        self.assertIn("OPTION3", error_msg)

    def test_optional_dict_value_success(self) -> None:
        """Test successful retrieval of optional dictionary values."""
        # Direct key
        value = self.config_manager.optional_dict_value(
            "subsection", {"default_key": "default_value"}
        )
        self.assertEqual(value, {"key3": "value3", "nested": {"key4": "value4"}})

        # Nested key
        value = self.config_manager.optional_dict_value(
            "subsection.nested", {"default_key": "default_value"}
        )
        self.assertEqual(value, {"key4": "value4"})

        # Empty section
        value = self.config_manager_no_prefix.optional_dict_value(
            "empty_section", {"default_key": "default_value"}
        )
        self.assertEqual(value, {})

        # None value (should return default)
        value = self.config_manager.optional_dict_value(
            "nonexistent", {"default_key": "default_value"}
        )
        self.assertEqual(value, {"default_key": "default_value"})

    def test_optional_dict_value_failure(self) -> None:
        """Test error cases for optional_dict_value method."""
        # Value is not a dictionary
        with self.assertRaises(TypeError) as context:
            self.config_manager_no_prefix.optional_dict_value(
                "section2.not_dict", {"default_key": "default_value"}
            )
        self.assertEqual(
            str(context.exception),
            "Value for property section2.not_dict must be a dictionary, got str",
        )

        # Non-existent key (should return default)
        value = self.config_manager.optional_dict_value(
            "nonexistent", {"default_key": "default_value"}
        )
        self.assertEqual(value, {"default_key": "default_value"})

        # None default
        value = self.config_manager.optional_dict_value("nonexistent", None)
        self.assertEqual(value, {})

        # Key exists but value is None (should return default)
        value = self.config_manager.optional_dict_value(
            "key2", {"default_key": "default_value"}
        )
        self.assertEqual(value, {"default_key": "default_value"})

    def test_full_key_with_prefix(self) -> None:
        """Test __full_key method with prefix."""
        # Access private method for testing
        full_key = self.config_manager._ConfigManager__full_key("test")
        self.assertEqual(full_key, "section1.test")

        # With nested path
        full_key = self.config_manager._ConfigManager__full_key("nested.path")
        self.assertEqual(full_key, "section1.nested.path")

    def test_full_key_without_prefix(self) -> None:
        """Test __full_key method without prefix."""
        # Empty prefix
        full_key = self.config_manager_no_prefix._ConfigManager__full_key("test")
        self.assertEqual(full_key, "test")

        # None prefix
        full_key = self.config_manager_none_prefix._ConfigManager__full_key("test")
        self.assertEqual(full_key, "test")

    def test_get_value_success(self) -> None:
        """Test successful value retrieval with __get_value."""
        # Simple key
        value = self.config_manager_no_prefix._ConfigManager__get_value("section1.key1")
        self.assertEqual(value, "value1")

        # Nested key
        value = self.config_manager_no_prefix._ConfigManager__get_value(
            "section1.subsection.key3"
        )
        self.assertEqual(value, "value3")

        # Empty section
        value = self.config_manager_no_prefix._ConfigManager__get_value("empty_section")
        self.assertEqual(value, {})

    def test_get_value_failures(self) -> None:
        """Test error cases for __get_value method."""
        # Non-existent key
        with self.assertRaises(KeyError) as context:
            self.config_manager_no_prefix._ConfigManager__get_value("nonexistent")
        self.assertEqual(
            str(context.exception),
            "\"Key 'nonexistent' not found in configuration path 'nonexistent'\"",
        )

        # Non-existent nested key
        with self.assertRaises(KeyError) as context:
            self.config_manager_no_prefix._ConfigManager__get_value(
                "section1.nonexistent"
            )
        self.assertEqual(
            str(context.exception),
            "\"Key 'nonexistent' not found in configuration path 'section1.nonexistent'\"",
        )

        # Parent is not a dictionary
        with self.assertRaises(TypeError) as context:
            self.config_manager_no_prefix._ConfigManager__get_value(
                "section2.not_dict.key"
            )
        self.assertEqual(
            str(context.exception),
            "Cannot access 'key' in 'section2.not_dict.key' as the parent is not a dictionary",
        )

    def test_integration_empty_config(self) -> None:
        """Test operation with an empty configuration."""
        empty_config = {}
        cm = ConfigManager(empty_config, "prefix")

        # Mandatory value with empty config
        with self.assertRaises(ValueError) as context:
            cm.mandatory_value("any_key", "Empty config error")
        self.assertEqual(str(context.exception), "Empty config error")

        # Optional value with empty config
        value = cm.optional_value("any_key", "default")
        self.assertEqual(value, "default")

        # Optional enum with empty config
        value = cm.optional_enum_value("any_key", TestEnum, TestEnum.OPTION3)
        self.assertEqual(value, TestEnum.OPTION3)

    def test_integration_realistic_scenario(self) -> None:
        """Test a realistic scenario with multiple operations."""
        app_config = {
            "app": {
                "name": "TestApp",
                "version": "1.0.0",
                "settings": {"debug": True, "timeout": 30, "mode": "OPTION2"},
            }
        }

        cm = ConfigManager(app_config, "app")

        # Get app name
        name = cm.mandatory_value("name", "App name is required")
        self.assertEqual(name, "TestApp")

        # Get optional setting with default
        cache_size = cm.optional_value("settings.cache_size", 1024)
        self.assertEqual(cache_size, 1024)

        # Get enum value
        mode = cm.optional_enum_value("settings.mode", TestEnum)
        self.assertEqual(mode, TestEnum.OPTION2)


if __name__ == "__main__":
    unittest.main()
