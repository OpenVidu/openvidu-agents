from typing import Any, Dict, TypeVar, Type, Optional
from enum import Enum

T = TypeVar("T", bound=Enum)


class ConfigManager:
    """
    A class to manage configuration values with support for mandatory and optional fields.

    This class provides methods to access configuration values in a dictionary-like structure,
    with support for nested paths using dot notation. It handles mandatory and optional values,
    provides type validation, and supports Enum validation.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        field_prefixes (str): The prefix to be added to the keys when accessing the configuration.
    """

    def __init__(self, config: Dict[str, Any], field_prefixes: str) -> None:
        """
        Initialize a ConfigManager with a configuration dictionary and optional field prefixes.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            field_prefixes (str): The prefix to be added to the keys when accessing the configuration.
                                 If None or empty, no prefix will be added.

        Raises:
            TypeError: If config is not a dictionary or field_prefixes is not a string or None.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if field_prefixes is not None and not isinstance(field_prefixes, str):
            raise TypeError("field_prefixes must be a string or None")
        self.config = config
        self.field_prefixes = field_prefixes or ""

    def mandatory_value(self, key: str, error_msg: str) -> Any:
        """
        Retrieve a mandatory configuration value.

        This method attempts to retrieve the value associated with the given key.
        If the value cannot be retrieved or is None, it raises a ValueError with the provided error message.

        Args:
            key (str): The key for the configuration value.
            error_msg (str): The error message to be used if the value cannot be retrieved.

        Returns:
            Any: The value associated with the given key.

        Raises:
            ValueError: If the value cannot be retrieved for any reason, with the provided error_msg.
        """
        try:
            value = self.__get_value(self.__full_key(key))
        except Exception:
            raise ValueError(error_msg)
        if value is None:
            raise ValueError(error_msg)
        return value

    def optional_value(self, key: str, default: Any) -> Any:
        """
        Retrieve an optional configuration value with a default fallback.

        This method attempts to retrieve the value associated with the given key.
        If the key does not exist, an exception occurs, or the value is None,
        it returns the provided default value.

        Args:
            key (str): The key to look up in the configuration.
            default (Any): The default value to return if the key is not found or the value is None.

        Returns:
            Any: The value associated with the key, or the default value if the key is not found
                 or the value is None.
        """
        try:
            value = self.__get_value(self.__full_key(key))
        except Exception:
            value = default
        if value is None:
            value = default
        return value
    
    def optional_string_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve an optional string value.

        This method retrieves a string value from the configuration.
        If the value is not a string, a TypeError is raised.
        If the value is None or the key is not found, the default value is returned.

        Args:
            key (str): The key to look up in the configuration.
            default (Optional[str], optional): The default value to return if the key is not found
                                            or the value is None. Defaults to None.

        Returns:
            Optional[str]: The string value associated with the key if it exists and is valid,
                        otherwise the default value.

        Raises:
            TypeError: If the value exists but it is not a string.
        """
        value = self.optional_value(key, None)
        if value is None:
            return default
        if not isinstance(value, str):
            raise TypeError(
                f"Value for property {self.__full_key(key)} must be a string, got {type(value).__name__}"
            )
        return value

    def optional_enum_value(
        self, key: str, enum_class: Type[T], default: Optional[T] = None
    ) -> Optional[T]:
        """
        Retrieve an optional enumeration value with validation.

        This method retrieves a value and validates it against an enumeration class.
        If the value exists but is not a valid enum member, a ValueError is raised.
        If the value is None or the key is not found, the default value is returned.

        Args:
            key (str): The key to look up in the configuration.
            enum_class (Type[T]): The enumeration class to validate the value against.
            default (Optional[T], optional): The default value to return if the key is not found
                                            or the value is None. Defaults to None.

        Returns:
            Optional[T]: The enum value associated with the key if it exists and is valid,
                        otherwise the default value.

        Raises:
            ValueError: If the value exists but it is not a valid member of the enumeration class.
        """
        value = self.optional_value(key, None)
        if value is None:
            return default
        try:
            return enum_class[value]
        except KeyError:
            valid_values = ", ".join(enum_class.__members__.keys())
            raise ValueError(
                f"Invalid value '{value}' for property {self.__full_key(key)}. Valid values are: {valid_values}"
            )

    def __full_key(self, key: str) -> str:
        """
        Construct a full key with the field_prefixes and the provided key.

        If field_prefixes is empty, the key is returned unchanged.
        Otherwise, the key is prefixed with field_prefixes followed by a dot.

        Args:
            key (str): The key to be prefixed.

        Returns:
            str: The prefixed key or the original key if field_prefixes is empty.
        """
        if not self.field_prefixes:
            return key
        return f"{self.field_prefixes}.{key}"

    def optional_dict_value(
        self, key: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve an optional dictionary value with validation.

        This method retrieves a value and validates that it is a dictionary.
        If the value exists but is not a dictionary, a TypeError is raised.
        If the value is None or the key is not found, the default value is returned.

        Args:
            key (str): The key to look up in the configuration.
            default (Optional[Dict[str, Any]], optional): The default dictionary to return if the key is not found
                                                         or the value is None. Defaults to None.

        Returns:
            Dict[str, Any]: The dictionary value associated with the key if it exists and is valid,
                           otherwise the default value.

        Raises:
            TypeError: If the value exists but it is not a dictionary.
        """
        value = self.optional_value(key, None)
        if value is None:
            return default if default is not None else {}

        if not isinstance(value, dict):
            raise TypeError(
                f"Value for property {self.__full_key(key)} must be a dictionary, got {type(value).__name__}"
            )

        return value

    def __get_value(self, field: str) -> Any:
        """
        Retrieve a value from the configuration based on a dot-separated field path.

        This method navigates through the configuration dictionary using the dot-separated
        field path and returns the value at the specified location.

        For example, given a field string "section1.subsection.key", it will return
        the value at config["section1"]["subsection"]["key"].

        Args:
            field (str): A dot-separated path to a value in the configuration.

        Returns:
            Any: The value from the configuration at the specified path.

        Raises:
            TypeError: If any part of the path except the last is not a dictionary.
            KeyError: If any part of the path does not exist in the configuration.
        """
        fields = field.split(".")
        value = self.config
        for f in fields:
            if not isinstance(value, dict):
                raise TypeError(
                    f"Cannot access '{f}' in '{field}' as the parent is not a dictionary"
                )
            try:
                value = value[f]
            except KeyError:
                raise KeyError(f"Key '{f}' not found in configuration path '{field}'")
        return value
