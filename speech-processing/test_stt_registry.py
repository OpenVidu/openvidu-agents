#!/usr/bin/env python3
"""
Test script to validate the STT provider registry.

This script demonstrates:
1. Registry is properly initialized at module load
2. All providers have implementation functions
3. Adding a new provider requires both registry entry and implementation function
"""

import unittest

import stt_impl
from stt_impl import STTProviderConfig


class TestSTTProviderRegistry(unittest.TestCase):
    """Test cases for the STT provider registry."""

    def test_registry_initialization(self):
        """Test that the registry initializes without errors."""
        # The import of stt_impl already validates initialization
        self.assertIsNotNone(stt_impl.STT_PROVIDERS)
        self.assertGreater(len(stt_impl.STT_PROVIDERS), 0)

        # Print provider list for visibility
        print(f"\nFound {len(stt_impl.STT_PROVIDERS)} registered providers:")
        for provider in sorted(stt_impl.STT_PROVIDERS.keys()):
            config = stt_impl.STT_PROVIDERS[provider]
            print(f"  - {provider:15} → {config.plugin_module}.{config.plugin_class}")

    def test_registry_completeness(self):
        """Test that all providers have implementation functions."""
        for provider, config in stt_impl.STT_PROVIDERS.items():
            with self.subTest(provider=provider):
                self.assertIsNotNone(
                    config.impl_function,
                    f"Provider '{provider}' has no implementation function",
                )
                self.assertIsNotNone(
                    config.plugin_module, f"Provider '{provider}' has no plugin module"
                )
                self.assertIsNotNone(
                    config.plugin_class, f"Provider '{provider}' has no plugin class"
                )

    def test_all_providers_have_valid_config(self):
        """Test that all provider configurations are valid."""
        for provider, config in stt_impl.STT_PROVIDERS.items():
            with self.subTest(provider=provider):
                # Check that plugin_module is a string
                self.assertIsInstance(
                    config.plugin_module,
                    str,
                    f"Provider '{provider}' plugin_module must be a string",
                )
                # Check that plugin_class is a string
                self.assertIsInstance(
                    config.plugin_class,
                    str,
                    f"Provider '{provider}' plugin_class must be a string",
                )
                # Check that impl_function is callable
                self.assertTrue(
                    callable(config.impl_function),
                    f"Provider '{provider}' impl_function must be callable",
                )

    def test_missing_implementation_detection(self):
        """Test that adding a provider without implementation raises an error."""
        # Save the current registry
        original_providers = stt_impl.STT_PROVIDERS.copy()

        try:
            # Add a fake provider to the registry
            stt_impl.STT_PROVIDERS["fake_provider"] = STTProviderConfig(
                impl_function=None,
                plugin_module="fake.module",
                plugin_class="FakeSTT",
            )

            # Try to initialize - should fail
            with self.assertRaises(RuntimeError) as context:
                stt_impl._initialize_stt_registry()

            self.assertIn(
                "Missing implementation functions",
                str(context.exception),
                "Should detect missing implementation function",
            )
            self.assertIn(
                "fake_provider",
                str(context.exception),
                "Error message should mention the fake provider",
            )
        finally:
            # Restore the original registry
            stt_impl.STT_PROVIDERS = original_providers
            stt_impl._initialize_stt_registry()

    def test_get_stt_impl_with_valid_provider(self):
        """Test that get_stt_impl works with the registry."""
        # Test with a valid provider name
        test_config = {
            "live_captions": {
                "provider": "aws",
                "aws": {
                    "aws_access_key_id": "test",
                    "aws_secret_access_key": "test",
                    "aws_default_region": "us-east-1",
                },
            }
        }

        # This will fail due to invalid credentials, but we're testing
        # that the registry lookup works correctly
        try:
            stt_impl.get_stt_impl(test_config)
        except ValueError as e:
            # Should NOT be "Unknown STT provider"
            self.assertNotIn(
                "Unknown STT provider",
                str(e),
                "Should not fail with unknown provider error",
            )
        except Exception:
            # Any other exception is fine - we just want to verify
            # the provider was found in the registry
            pass

    def test_get_stt_impl_with_invalid_provider(self):
        """Test that get_stt_impl raises error for unknown provider."""
        test_config = {"live_captions": {"provider": "nonexistent_provider"}}

        with self.assertRaises(ValueError) as context:
            stt_impl.get_stt_impl(test_config)

        self.assertIn(
            "Unknown STT provider",
            str(context.exception),
            "Should raise error for unknown provider",
        )
        self.assertIn(
            "nonexistent_provider",
            str(context.exception),
            "Error message should mention the invalid provider",
        )

    def test_get_stt_impl_missing_provider_config(self):
        """Test that get_stt_impl raises error when provider is not configured."""
        test_config = {"live_captions": {}}

        with self.assertRaises(ValueError) as context:
            stt_impl.get_stt_impl(test_config)

        self.assertIn(
            "provider not defined",
            str(context.exception),
            "Should raise error when provider is not configured",
        )

    def test_provider_count(self):
        """Test that we have the expected number of providers."""
        # Update this number when adding new providers
        expected_count = 21
        actual_count = len(stt_impl.STT_PROVIDERS)

        self.assertEqual(
            actual_count,
            expected_count,
            f"Expected {expected_count} providers, found {actual_count}. "
            f"If you added a new provider, update this test.",
        )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
