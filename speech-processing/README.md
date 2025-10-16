# speech-processing

AI services for transcribing, translating and summarizing video conference conversations. See [https://openvidu.io/latest/docs/agents/speech-processing/].

## STT Provider Implementation Guide

This guide explains how to add a new STT provider to the speech-processing agent.

### Overview

The STT provider system uses a **centralized registry pattern** that ensures:
- Single source of truth for all providers
- Automatic validation at boot time
- No duplication of provider lists across the codebase

### Architecture

#### 1. Registry (`STT_PROVIDERS`)

Located at the top of `stt_impl.py`, this dictionary defines all supported providers:

```python
STT_PROVIDERS = {
    "provider_name": STTProviderConfig(
        impl_function=None,  # Set during initialization
        plugin_module="livekit.plugins.provider",
        plugin_class="STT",
    ),
}
```

#### 2. Implementation Functions

Each provider requires a `get_<provider>_stt_impl(agent_config)` function that:
- Validates configuration
- Creates and returns the STT instance

#### 3. Automatic Validation

At module load time, `_initialize_stt_registry()` validates that:
- Every registry entry has an implementation function
- Every implementation function has a registry entry

**If validation fails, the application won't start.**

### Adding a New Provider

#### Step 1: Add to Registry

Add your provider to the `STT_PROVIDERS` dictionary in `stt_impl.py`:

```python
STT_PROVIDERS = {
    # ... existing providers ...
    "newprovider": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.newprovider",
        plugin_class="STT",  # or custom class name
    ),
}
```

**Configuration:**
- `plugin_module`: The Python module path for the plugin
- `plugin_class`: The class name (usually "STT", but can vary like "WizperSTT" for FAL)

#### Step 2: Implement the Function

Create the implementation function in `stt_impl.py`:

```python
def get_newprovider_stt_impl(agent_config) -> stt.STT:
    from livekit.plugins import newprovider

    config_manager = ConfigManager(agent_config, "live_captions.newprovider")
    
    # Validate required credentials
    api_key = config_manager.mandatory_value(
        "api_key",
        "Wrong NewProvider credentials. live_captions.newprovider.api_key must be set"
    )
    
    # Get optional parameters
    language = config_manager.configured_string_value("language")
    model = config_manager.configured_string_value("model")
    
    # Build kwargs, excluding NOT_PROVIDED values
    kwargs = {
        k: v
        for k, v in {
            "language": language,
            "model": model,
        }.items()
        if v is not NOT_PROVIDED
    }
    
    # Return configured STT instance
    return newprovider.STT(api_key=api_key, **kwargs)
```

#### Step 3: Update Registry Initialization

Add your implementation function to the `provider_impl_map` in `_initialize_stt_registry()`:

```python
def _initialize_stt_registry():
    provider_impl_map = {
        # ... existing providers ...
        "newprovider": get_newprovider_stt_impl,
    }
    # ... rest of function ...
```

#### Step 4: Add Configuration Schema

Add the configuration section to `agent-speech-processing.yaml`:

```yaml
live_captions:
  provider: # ... existing providers or newprovider
  
  # ... existing provider configs ...
  
  newprovider:
    # API key for NewProvider. See https://newprovider.com/api-keys
    api_key:
    # The language code to use for transcription (e.g., "en" for English)
    language:
    # The model to use for transcription
    model:
```

#### Step 5: Install Dependencies

Add the plugin to `requirements.txt`:

```txt
livekit-agents[silero,turn_detector,...,newprovider]==1.2.14
```

Or install separately:

```bash
pip install livekit-plugins-newprovider
```

#### Step 6: Test

Run the validation script:

```bash
python3 test_stt_registry.py
```

This will verify:
- Registry is properly initialized
- All providers have implementation functions
- No orphaned implementations

### What Happens if You Forget Something?

#### If you add to registry but forget the implementation:

```
RuntimeError: Missing implementation functions for STT providers: newprovider. 
Please implement get_{provider}_stt_impl() functions for these providers.
```

#### If you add implementation but forget the registry:

```
RuntimeError: Implementation functions exist for unregistered STT providers: newprovider. 
Please add these providers to the STT_PROVIDERS registry.
```

#### If you use an unknown provider at runtime:

```
ValueError: Unknown STT provider: newprovider. 
Supported providers: assemblyai, aws, azure, ...
```

### Benefits of This Approach

1. **Single Source of Truth**: Provider list is defined once in `STT_PROVIDERS`
2. **Boot-time Validation**: Errors are caught immediately when the module loads
3. **Type Safety**: Registry includes plugin module and class information
4. **Maintainability**: Adding a provider is a clear, documented process. Upgrading livekit-plugins is safe, as breaking changes will be caught
5. **Auto-discovery**: Language defaults are automatically discovered from plugin constructors
6. **No Duplication**: No need for long if-elif chains or multiple provider lists

### Example: Real Implementation

See any existing provider like `get_aws_stt_impl()`, `get_openai_stt_impl()`, etc. for complete examples.
