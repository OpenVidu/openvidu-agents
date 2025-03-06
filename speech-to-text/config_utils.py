class ConfigManager:
    """
    A class to manage configuration values with support for mandatory and optional fields.

    Attributes:
        config (dict): The configuration dictionary.
        field_prefixes (str): The prefix to be added to the keys when accessing the configuration.

    Methods:
        mandatory_value(key, errorMsg):
            Retrieves the value for a mandatory configuration key. Raises a ValueError if the key is not found.

        optional_value(key, default):
            Retrieves the value for an optional configuration key. Returns the default value if the key is not found.

        _getValue(field):
            Retrieves the value of a nested field in the configuration dictionary.
    """

    def __init__(self, config, field_prefixes):
        self.config = config
        self.field_prefixes = field_prefixes

    def mandatory_value(self, key, errorMsg):
        """
        Retrieve a mandatory configuration value.

        This method attempts to retrieve the value associated with the given key.
        If the value cannot be retrieved, it raises a ValueError with the provided error message.

        Args:
            key (str): The key for the configuration value.
            errorMsg (str): The error message to be used if the value cannot be retrieved.

        Returns:
            The value associated with the given key.

        Raises:
            ValueError: If the value cannot be retrieved.
        """
        try:
            full_key = self.field_prefixes + "." + key
            value = self._getValue(full_key)
        except Exception:
            raise ValueError(errorMsg)
        return value

    def optional_value(self, key, default):
        """
        Retrieve the value associated with the given key from the configuration.
        If the key does not exist or the value is None, return the provided default value.

        Args:
            key (str): The key to look up in the configuration.
            default: The default value to return if the key is not found or the value is None.

        Returns:
            The value associated with the key, or the default value if the key is not found or the value is None.
        """
        try:
            full_key = self.field_prefixes + "." + key
            value = self.config[full_key]
        except Exception:
            value = default
        if value is None:
            value = default
        return value

    def _getValue(self, field):
        """
        Retrieve the value from the configuration dictionary based on a dot-separated field path.
        Given a string "field1.field2.field3" returns the value of the field3 in the config object,
        accessing it like: config["field1"]["field2"]["field3"]

        Args:
            field (str): A dot-separated string representing the path to the desired value in the configuration dictionary.

        Returns:
            The value from the configuration dictionary corresponding to the specified field path.

        Raises:
            KeyError: If any part of the field path does not exist in the configuration dictionary.
        """
        fields = field.split(".")
        value = self.config
        for f in fields:
            value = value[f]
        return value
