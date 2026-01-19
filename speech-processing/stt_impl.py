import logging
import os
import json
import tempfile
import inspect
from typing import Callable, NamedTuple
from livekit.agents import stt
from livekit.agents.types import NotGivenOr, NotGiven
from livekit.agents.voice.agent_session import TurnDetectionMode

from openviduagentutils.config_manager import ConfigManager
from openviduagentutils.not_provided import NOT_PROVIDED

# Import all available LiveKit plugins at module level (main thread) to ensure proper registration

# Track which plugins are available in this container
AVAILABLE_PLUGINS = {}

# Try importing each plugin - if it fails, mark as unavailable
def _try_import_plugin(plugin_name: str):
    """Try to import a plugin module. Returns the module if successful, None otherwise."""
    try:
        module = __import__(f"livekit.plugins.{plugin_name}", fromlist=[plugin_name])
        AVAILABLE_PLUGINS[plugin_name] = module
        return module
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning(f"Plugin '{plugin_name}' not available in this container: {e}")
        return None

# Import all potentially available plugins
aws = _try_import_plugin("aws")
azure = _try_import_plugin("azure")
openai = _try_import_plugin("openai")
google = _try_import_plugin("google")
groq = _try_import_plugin("groq")
deepgram = _try_import_plugin("deepgram")
assemblyai = _try_import_plugin("assemblyai")
fal = _try_import_plugin("fal")
clova = _try_import_plugin("clova")
speechmatics = _try_import_plugin("speechmatics")
gladia = _try_import_plugin("gladia")
sarvam = _try_import_plugin("sarvam")
spitch = _try_import_plugin("spitch")
mistralai = _try_import_plugin("mistralai")
cartesia = _try_import_plugin("cartesia")
soniox = _try_import_plugin("soniox")
nvidia = _try_import_plugin("nvidia")
vosk = _try_import_plugin("vosk")


# STT Provider Registry
class STTProviderConfig(NamedTuple):
    """Configuration for an STT provider."""

    impl_function: Callable[[dict], stt.STT]
    plugin_module: str
    plugin_class: str


# Central registry of all supported STT providers
STT_PROVIDERS = {
    "aws": STTProviderConfig(
        impl_function=None,  # Will be set after function definitions
        plugin_module="livekit.plugins.aws",
        plugin_class="STT",
    ),
    "azure": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.azure",
        plugin_class="STT",
    ),
    "azure_openai": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.openai",
        plugin_class="STT",
    ),
    "google": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.google",
        plugin_class="STT",
    ),
    "openai": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.openai",
        plugin_class="STT",
    ),
    "groq": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.groq",
        plugin_class="STT",
    ),
    "deepgram": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.deepgram",
        plugin_class="STT",
    ),
    "assemblyai": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.assemblyai",
        plugin_class="STT",
    ),
    "fal": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.fal",
        plugin_class="WizperSTT",
    ),
    "clova": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.clova",
        plugin_class="STT",
    ),
    "speechmatics": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.speechmatics",
        plugin_class="STT",
    ),
    "gladia": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.gladia",
        plugin_class="STT",
    ),
    "sarvam": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.sarvam",
        plugin_class="STT",
    ),
    "mistralai": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.mistralai",
        plugin_class="STT",
    ),
    "cartesia": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.cartesia",
        plugin_class="STT",
    ),
    "soniox": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.soniox",
        plugin_class="STT",
    ),
    "spitch": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.spitch",
        plugin_class="STT",
    ),
    "nvidia": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.nvidia",
        plugin_class="STT",
    ),
    "vosk": STTProviderConfig(
        impl_function=None,
        plugin_module="livekit.plugins.vosk",
        plugin_class="STT",
    ),
}

# https://docs.livekit.io/agents/build/turns/turn-detector/#detection-accuracy
SUPPORTED_LANGUAGES_IN_MULTILINGUAL_TURN_DETECTION = [
    "hi",  # Hindi
    "ko",  # Korean
    "fr",  # French
    "pt",  # Portuguese
    "id",  # Indonesian
    "ru",  # Russian
    "en",  # English
    "zh",  # Chinese
    "ja",  # Japanese
    "it",  # Italian
    "es",  # Spanish
    "de",  # German
    "tr",  # Turkish
    "nl",  # Dutch
]


def get_aws_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.aws")
    wrong_credentials = "Wrong AWS credentials. live_captions.aws.aws_access_key_id, live_captions.aws.aws_secret_access_key and live_captions.aws.aws_default_region must be set"

    try:
        api_key = config_manager.mandatory_value("aws_access_key_id", wrong_credentials)
        api_secret = config_manager.mandatory_value(
            "aws_secret_access_key", wrong_credentials
        )
        default_region = config_manager.mandatory_value(
            "aws_default_region", wrong_credentials
        )

        # FIX for: https://github.com/awslabs/amazon-transcribe-streaming-sdk/issues/7#issuecomment-1677230478
        os.environ["AWS_ACCESS_KEY_ID"] = api_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = api_secret
        os.environ["AWS_DEFAULT_REGION"] = default_region
    except Exception:
        raise ValueError(wrong_credentials)
    if api_key is None or api_secret is None or default_region is None:
        raise ValueError(wrong_credentials)

    language = config_manager.configured_string_value("language")
    vocabulary_name = config_manager.configured_string_value("vocabulary_name")
    language_model_name = config_manager.configured_string_value("language_model_name")
    enable_partial_results_stabilization = config_manager.configured_boolean_value(
        "enable_partial_results_stabilization"
    )
    partial_results_stability = config_manager.configured_string_value(
        "partial_results_stability"
    )
    vocab_filter_name = config_manager.configured_string_value("vocab_filter_name")
    vocab_filter_method = config_manager.configured_string_value("vocab_filter_method")

    kwargs = {
        k: v
        for k, v in {
            "region": default_region,
            "language": language,
            "vocabulary_name": vocabulary_name,
            "language_model_name": language_model_name,
            "enable_partial_results_stabilization": enable_partial_results_stabilization,
            "partial_results_stability": partial_results_stability,
            "vocab_filter_name": vocab_filter_name,
            "vocab_filter_method": vocab_filter_method,
        }.items()
        if v is not NOT_PROVIDED
    }

    return aws.STT(**kwargs)


def get_azure_stt_impl(agent_config) -> stt.STT:
    from azure.cognitiveservices.speech.enums import ProfanityOption

    config_manager = ConfigManager(agent_config, "live_captions.azure")
    wrong_credentials = "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n    - speech_auth_token + speech_region"

    speech_host = config_manager.configured_string_value("speech_host")
    speech_region = config_manager.configured_string_value("speech_region")
    speech_key = config_manager.configured_string_value("speech_key")
    speech_auth_token = config_manager.configured_string_value("speech_auth_token")
    language = config_manager.configured_value("language")
    profanity = config_manager.configured_enum_value("profanity", ProfanityOption)
    phrase_list = config_manager.configured_list_value("phrase_list", str)
    explicit_punctuation = config_manager.configured_boolean_value(
        "explicit_punctuation"
    )

    has_host = speech_host is not NOT_PROVIDED
    has_key_region = (speech_key is not NOT_PROVIDED) and (
        speech_region is not NOT_PROVIDED
    )
    has_token_region = (speech_auth_token is not NOT_PROVIDED) and (
        speech_region is not NOT_PROVIDED
    )
    if not (has_host or has_key_region or has_token_region):
        raise ValueError(wrong_credentials)

    kwargs = {
        k: v
        for k, v in {
            "speech_host": speech_host,
            "speech_region": speech_region,
            "speech_key": speech_key,
            "speech_auth_token": speech_auth_token,
            "language": language,
            "profanity": profanity,
            "phrase_list": phrase_list,
            "explicit_punctuation": explicit_punctuation,
        }.items()
        if v is not NOT_PROVIDED
    }

    return azure.STT(
        **kwargs,
    )


def get_azure_openai_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.azure_openai")

    azure_api_key = config_manager.mandatory_value(
        "azure_api_key",
        "Wrong Azure OpenAI credentials. live_captions.azure_openai.azure_api_key must be set",
    )
    azure_endpoint = config_manager.mandatory_value(
        "azure_endpoint",
        "Wrong Azure OpenAI configuration. live_captions.azure_openai.azure_endpoint must be set",
    )
    azure_ad_token = config_manager.configured_string_value("azure_ad_token")
    api_version = config_manager.configured_string_value("api_version")
    azure_deployment = config_manager.configured_string_value("azure_deployment")
    organization = config_manager.configured_string_value("organization")
    project = config_manager.configured_string_value("project")
    language = config_manager.configured_string_value("language")
    detect_language = config_manager.configured_boolean_value("detect_language")
    model = config_manager.configured_string_value("model")
    prompt = config_manager.configured_string_value("prompt")

    kwargs = {
        k: v
        for k, v in {
            "azure_ad_token": azure_ad_token,
            "api_version": api_version,
            "azure_deployment": azure_deployment,
            "organization": organization,
            "project": project,
            "language": language,
            "detect_language": detect_language,
            "model": model,
            "prompt": prompt,
        }.items()
        if v is not NOT_PROVIDED
    }

    return openai.STT.with_azure(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        **kwargs,
    )


def get_google_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.google")
    wrong_credentials = (
        "Wrong Google credentials. live_captions.google.credentials_info must be set"
    )

    credentials_info_str = config_manager.mandatory_value(
        "credentials_info", wrong_credentials
    )
    try:
        credentials_info = json.loads(credentials_info_str)
    except Exception as jsonerror:
        raise ValueError(wrong_credentials + " and must be a valid JSON", jsonerror)
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.write(json.dumps(credentials_info).encode())
        temp_file.close()
    except Exception as e:
        raise ValueError(
            "Failed to create a temporary JSON file for google credentials", e
        )
    model = config_manager.configured_string_value("model")
    languages = config_manager.configured_string_value("languages")
    detect_language = config_manager.configured_boolean_value("detect_language")
    location = config_manager.configured_string_value("location")
    punctuate = config_manager.configured_boolean_value("punctuate")
    spoken_punctuation = config_manager.configured_boolean_value("spoken_punctuation")
    interim_results = config_manager.configured_boolean_value("interim_results")

    kwargs = {
        k: v
        for k, v in {
            "model": model,
            "languages": languages,
            "detect_language": detect_language,
            "location": location,
            "punctuate": punctuate,
            "spoken_punctuation": spoken_punctuation,
            "interim_results": interim_results,
        }.items()
        if v is not NOT_PROVIDED
    }

    return google.STT(
        credentials_file=temp_file.name,
        **kwargs,
    )


def get_openai_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.openai")
    wrong_credentials = (
        "Wrong OpenAI credentials. live_captions.openai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.configured_string_value("model")
    language = config_manager.configured_string_value("language")
    detect_language = config_manager.configured_boolean_value("detect_language")
    prompt = config_manager.configured_string_value("prompt")

    kwargs = {
        k: v
        for k, v in {
            "model": model,
            "language": language,
            "prompt": prompt,
            "detect_language": detect_language,
        }.items()
        if v is not NOT_PROVIDED
    }

    return openai.STT(api_key=api_key, **kwargs)


def get_groq_stt_impl(agent_config):
    config_manager = ConfigManager(agent_config, "live_captions.groq")
    wrong_credentials = "Wrong Groq credentials. live_captions.groq.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.configured_string_value("model")
    language = config_manager.configured_string_value("language")
    detect_language = config_manager.configured_boolean_value("detect_language")
    prompt = config_manager.configured_string_value("prompt")
    base_url = config_manager.configured_string_value("base_url")

    kwargs = {
        k: v
        for k, v in {
            "model": model,
            "language": language,
            "detect_language": detect_language,
            "prompt": prompt,
            "base_url": base_url,
        }.items()
        if v is not NOT_PROVIDED
    }

    return groq.STT(api_key=api_key, **kwargs)


def get_deepgram_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.deepgram")
    wrong_credentials = (
        "Wrong Deepgram credentials. live_captions.deepgram.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.configured_string_value("model")
    language = config_manager.configured_string_value("language")
    detect_language = config_manager.configured_boolean_value("detect_language")
    interim_results = config_manager.configured_boolean_value("interim_results")
    smart_format = config_manager.configured_boolean_value("smart_format")
    no_delay = config_manager.configured_boolean_value("no_delay")
    punctuate = config_manager.configured_boolean_value("punctuate")
    filler_words = config_manager.configured_boolean_value("filler_words")
    profanity_filter = config_manager.configured_boolean_value("profanity_filter")
    numerals = config_manager.configured_boolean_value("numerals")
    keywords = config_manager.configured_list_value("keywords")
    keyterms = config_manager.configured_list_value("keyterms", str)

    kwargs = {
        k: v
        for k, v in {
            "model": model,
            "language": language,
            "detect_language": detect_language,
            "interim_results": interim_results,
            "smart_format": smart_format,
            "no_delay": no_delay,
            "punctuate": punctuate,
            "filler_words": filler_words,
            "profanity_filter": profanity_filter,
            "numerals": numerals,
            "keywords": keywords,
            "keyterms": keyterms,
        }.items()
        if v is not NOT_PROVIDED
    }

    return deepgram.STT(
        api_key=api_key,
        **kwargs,
    )


def get_assemblyai_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.assemblyai")
    wrong_credentials = (
        "Wrong AssemblyAI credentials. live_captions.assemblyai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    end_of_turn_confidence_threshold = config_manager.configured_numeric_value(
        "end_of_turn_confidence_threshold"
    )
    min_end_of_turn_silence_when_confident = config_manager.configured_numeric_value(
        "min_end_of_turn_silence_when_confident"
    )
    max_turn_silence = config_manager.configured_numeric_value("max_turn_silence")
    format_turns = config_manager.configured_boolean_value("format_turns")
    keyterms_prompt = config_manager.configured_list_value("keyterms_prompt", str)

    kwargs = {
        k: v
        for k, v in {
            "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
            "min_end_of_turn_silence_when_confident": min_end_of_turn_silence_when_confident,
            "max_turn_silence": max_turn_silence,
            "format_turns": format_turns,
            "keyterms_prompt": keyterms_prompt,
        }.items()
        if v is not NOT_PROVIDED
    }

    return assemblyai.STT(api_key=api_key, **kwargs)


def get_fal_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.fal")
    wrong_credentials = "Wrong FAL credentials. live_captions.fal.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.configured_string_value("language")

    kwargs = {k: v for k, v in {"language": language}.items() if v is not NOT_PROVIDED}

    return fal.WizperSTT(api_key=api_key, **kwargs)


def get_clova_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.clova")
    wrong_credentials = "Wrong Clova credentials. live_captions.clova.api_key and live_captions.clova.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    invoke_url = config_manager.mandatory_value("invoke_url", wrong_credentials)
    language = config_manager.configured_string_value("language")
    threshold = config_manager.configured_numeric_value("threshold")

    kwargs = {
        k: v
        for k, v in {"language": language, "threshold": threshold}.items()
        if v is not NOT_PROVIDED
    }

    return clova.STT(invoke_url=invoke_url, secret=api_key, **kwargs)


def get_speechmatics_stt_impl(agent_config) -> stt.STT:
    from livekit.plugins.speechmatics.types import TranscriptionConfig

    config_manager = ConfigManager(agent_config, "live_captions.speechmatics")
    wrong_credentials = (
        "Wrong Speechmatics credentials. live_captions.speechmatics.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "en")
    operating_point = config_manager.optional_string_value(
        "operating_point", "enhanced"
    )
    enable_partials = config_manager.configured_boolean_value("enable_partials")
    output_locale = config_manager.configured_string_value("output_locale")
    max_delay = config_manager.configured_numeric_value("max_delay")
    max_delay_mode = config_manager.configured_string_value("max_delay_mode")
    punctuation_overrides = config_manager.configured_dict_value(
        "punctuation_overrides"
    )
    additional_vocab = config_manager.configured_list_value("additional_vocab")
    speaker_diarization_config = config_manager.configured_dict_value(
        "speaker_diarization_config"
    )

    kwargs = {
        k: v
        for k, v in {
            "language": language,
            "operating_point": operating_point,
            "output_locale": output_locale,
            "enable_partials": enable_partials,
            "max_delay": max_delay,
            "max_delay_mode": max_delay_mode,
            "punctuation_overrides": punctuation_overrides,
            "additional_vocab": additional_vocab,
            "speaker_diarization_config": speaker_diarization_config,
        }.items()
        if v is not NOT_PROVIDED
    }

    return speechmatics.STT(
        api_key=api_key, transcription_config=TranscriptionConfig(**kwargs)
    )


def get_gladia_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.gladia")
    wrong_credentials = (
        "Wrong Gladia credentials. live_captions.gladia.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    interim_results = config_manager.configured_boolean_value("interim_results")
    languages = config_manager.configured_list_value("languages", str)
    code_switching = config_manager.configured_boolean_value("code_switching")
    pre_processing_audio_enhancer = config_manager.configured_boolean_value(
        "pre_processing_audio_enhancer"
    )
    pre_processing_speech_threshold = config_manager.configured_numeric_value(
        "pre_processing_speech_threshold"
    )

    kwargs = {
        k: v
        for k, v in {
            "interim_results": interim_results,
            "languages": languages,
            "code_switching": code_switching,
            "pre_processing_audio_enhancer": pre_processing_audio_enhancer,
            "pre_processing_speech_threshold": pre_processing_speech_threshold,
        }.items()
        if v is not NOT_PROVIDED
    }

    return gladia.STT(api_key=api_key, **kwargs)


def get_sarvam_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.sarvam")
    wrong_credentials = (
        "Wrong Sarvam credentials. live_captions.sarvam.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "unknown")
    model = config_manager.configured_string_value("model")

    kwargs = {k: v for k, v in {"model": model}.items() if v is not NOT_PROVIDED}

    return sarvam.STT(
        api_key=api_key,
        language=language,
        **kwargs,
    )


def get_spitch_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.spitch")
    wrong_credentials = (
        "Wrong Spitch credentials. live_captions.spitch.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "en")

    # livekit-plugins-spitch require the SPITCH_API_KEY env var to be set
    os.environ["SPITCH_API_KEY"] = api_key

    kwargs = {k: v for k, v in {"language": language}.items() if v is not NOT_PROVIDED}

    return spitch.STT(**kwargs)


def get_vosk_stt_impl(agent_config) -> stt.STT:
    # Direct mapping from Vosk model names to language codes
    # Based on pre-installed models in the container
    VOSK_MODEL_TO_LANGUAGE = {
        "vosk-model-en-us-0.22-lgraph": "en-US",
        "vosk-model-small-cn-0.22": "zh-CN",
        "vosk-model-small-de-0.15": "de",
        "vosk-model-small-en-in-0.4": "en-IN",
        "vosk-model-small-es-0.42": "es",
        "vosk-model-small-fr-0.22": "fr",
        "vosk-model-small-hi-0.22": "hi",
        "vosk-model-small-it-0.22": "it",
        "vosk-model-small-ja-0.22": "ja",
        "vosk-model-small-nl-0.22": "nl",
        "vosk-model-small-pt-0.3": "pt",
        "vosk-model-small-ru-0.22": "ru",
    }

    config_manager = ConfigManager(agent_config, "live_captions.vosk")

    model = config_manager.configured_string_value("model")
    language = config_manager.configured_string_value("language")
    sample_rate = config_manager.configured_numeric_value("sample_rate")
    partial_results = config_manager.configured_boolean_value("partial_results")

    # Auto-detect language from model name if not provided
    if language is NOT_PROVIDED and model is not NOT_PROVIDED:
        detected_language = VOSK_MODEL_TO_LANGUAGE.get(model)
        if detected_language:
            logging.info(
                f"Auto-detected language '{detected_language}' from Vosk model '{model}'"
            )
            language = detected_language
        else:
            logging.warning(
                f"Could not auto-detect language from Vosk model '{model}', defaulting to 'en-US'"
            )
            language = "en-US"
    elif language is NOT_PROVIDED:
        language = "en-US"

    kwargs = {
        k: v
        for k, v in {
            "model_path": (
                "vosk-models/" + model if model is not NOT_PROVIDED else NOT_PROVIDED
            ),
            "language": language,
            "sample_rate": sample_rate,
            "partial_results": partial_results,
        }.items()
        if v is not NOT_PROVIDED
    }

    return vosk.STT(**kwargs)


def get_mistralai_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.mistralai")
    wrong_credentials = (
        "Wrong MistralAI credentials. live_captions.mistralai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)

    language = config_manager.configured_string_value("language")
    model = config_manager.configured_string_value("model")

    kwargs = {
        k: v
        for k, v in {"language": language, "model": model}.items()
        if v is not NOT_PROVIDED
    }

    return mistralai.STT(api_key=api_key, **kwargs)


def get_cartesia_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.cartesia")
    wrong_credentials = (
        "Wrong Cartesia credentials. live_captions.cartesia.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)

    language = config_manager.configured_string_value("language")
    model = config_manager.configured_string_value("model")

    kwargs = {
        k: v
        for k, v in {"language": language, "model": model}.items()
        if v is not NOT_PROVIDED
    }

    return cartesia.STT(api_key=api_key, **kwargs)


def get_soniox_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.soniox")
    wrong_credentials = (
        "Wrong Soniox credentials. live_captions.soniox.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)

    model = config_manager.configured_string_value("model")
    language_hints = config_manager.configured_list_value("language_hints", str)
    context = config_manager.configured_string_value("context")

    kwargs = {
        k: v
        for k, v in {
            "model": model,
            "language_hints": language_hints,
            "context": context,
        }.items()
        if v is not NOT_PROVIDED
    }

    return soniox.STT(api_key=api_key, params=soniox.STTOptions(**kwargs))


def get_nvidia_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.nvidia")

    api_key = config_manager.configured_string_value("api_key")
    server = config_manager.configured_string_value("server")
    use_ssl = config_manager.configured_boolean_value("use_ssl")

    # If server is defined, it takes precedence (self-hosted scenario)
    # If api_key is empty and server is not defined, raise an error
    if api_key is NOT_PROVIDED and server is NOT_PROVIDED:
        raise ValueError(
            "Wrong NVIDIA configuration. Either live_captions.nvidia.api_key must be set "
            "(for NVIDIA cloud) or live_captions.nvidia.server must be set (for self-hosted Riva NIM)."
        )

    model = config_manager.configured_string_value("model")
    function_id = config_manager.configured_string_value("function_id")
    punctuate = config_manager.configured_boolean_value("punctuate")
    language_code = config_manager.configured_string_value("language_code")
    sample_rate = config_manager.configured_numeric_value("sample_rate")

    kwargs = {
        k: v
        for k, v in {
            "api_key": api_key,
            "model": model,
            "function_id": function_id,
            "punctuate": punctuate,
            "language_code": language_code,
            "sample_rate": sample_rate,
            "server": server,
            "use_ssl": use_ssl,
        }.items()
        if v is not NOT_PROVIDED
    }

    return nvidia.STT(**kwargs)


# Initialize the registry with implementation functions
def _initialize_stt_registry():
    """Initialize the STT provider registry with implementation functions.

    This must be called after all get_*_stt_impl functions are defined.
    It validates that all providers have their implementation functions.
    """
    global STT_PROVIDERS

    provider_impl_map = {
        "aws": get_aws_stt_impl,
        "azure": get_azure_stt_impl,
        "azure_openai": get_azure_openai_stt_impl,
        "google": get_google_stt_impl,
        "openai": get_openai_stt_impl,
        "groq": get_groq_stt_impl,
        "deepgram": get_deepgram_stt_impl,
        "assemblyai": get_assemblyai_stt_impl,
        "fal": get_fal_stt_impl,
        "clova": get_clova_stt_impl,
        "speechmatics": get_speechmatics_stt_impl,
        "gladia": get_gladia_stt_impl,
        "sarvam": get_sarvam_stt_impl,
        "mistralai": get_mistralai_stt_impl,
        "cartesia": get_cartesia_stt_impl,
        "soniox": get_soniox_stt_impl,
        "nvidia": get_nvidia_stt_impl,
        "spitch": get_spitch_stt_impl,
        "vosk": get_vosk_stt_impl,
    }

    # Validate that all registered providers have implementation functions
    missing_impls = []
    for provider_name in STT_PROVIDERS.keys():
        if provider_name not in provider_impl_map:
            missing_impls.append(provider_name)

    if missing_impls:
        raise RuntimeError(
            f"Missing implementation functions for STT providers: {', '.join(missing_impls)}. "
            f"Please implement get_{{provider}}_stt_impl() functions for these providers."
        )

    # Validate that all implementation functions are registered
    extra_impls = []
    for provider_name in provider_impl_map.keys():
        if provider_name not in STT_PROVIDERS:
            extra_impls.append(provider_name)

    if extra_impls:
        raise RuntimeError(
            f"Implementation functions exist for unregistered STT providers: {', '.join(extra_impls)}. "
            f"Please add these providers to the STT_PROVIDERS registry."
        )

    # Update the registry with implementation functions
    STT_PROVIDERS = {
        provider: STTProviderConfig(
            impl_function=provider_impl_map[provider],
            plugin_module=config.plugin_module,
            plugin_class=config.plugin_class,
        )
        for provider, config in STT_PROVIDERS.items()
    }


def get_stt_impl(agent_config) -> stt.STT:
    """Get the STT implementation for the configured provider.

    Args:
        agent_config: Agent configuration dictionary

    Returns:
        Configured STT instance

    Raises:
        ValueError: If provider is not configured or unknown
    """
    try:
        stt_provider = agent_config["live_captions"]["provider"]
    except Exception:
        stt_provider = None

    if stt_provider is None:
        raise ValueError("live_captions.provider not defined in agent configuration")

    if stt_provider not in STT_PROVIDERS:
        raise ValueError(
            f"Unknown STT provider: {stt_provider}. "
            f"Supported providers: {', '.join(sorted(STT_PROVIDERS.keys()))}"
        )

    # Check if the plugin is available in this container
    provider_config = STT_PROVIDERS[stt_provider]
    plugin_name = provider_config.plugin_module.split(".")[-1]  # Extract plugin name from module path
    
    if plugin_name not in AVAILABLE_PLUGINS:
        available_providers = [
            name for name, config in STT_PROVIDERS.items()
            if config.plugin_module.split(".")[-1] in AVAILABLE_PLUGINS
        ]
        raise ValueError(
            f"STT provider '{stt_provider}' is not available in this container. "
            f"The required plugin '{plugin_name}' is not installed. "
            f"Available providers in this container: {', '.join(sorted(available_providers)) or 'none'}. "
            f"Please use the correct Docker image for your desired provider "
            f"(e.g., openvidu/agent-speech-processing-cloud:main for cloud providers, "
            f"openvidu/agent-speech-processing-vosk:main for vosk)."
        )

    logging.info(f"Using {stt_provider} as STT provider")

    return provider_config.impl_function(agent_config)


def get_best_turn_detector(agent_config) -> NotGivenOr[TurnDetectionMode]:
    """Get the best turn detection mode for the given agent configuration.

    The best turn detection mode is determined by the following rules:
        1. For STT providers that support native turn detection, use "stt" (for now only "assemblyai") (https://docs.livekit.io/agents/build/turns/#stt-endpointing)
        2. Determine the language in use by the model, if defined. Try to get the "language" from the agent config, or determine it from the default language configuration of each stt plugin.
            1. If the language is any variant of English (en, en-US, en-GB, etc), use livekit.plugins.turn_detector.english.EnglishModel
            2. If the language is any variant of any of these languages [Spanish, French, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Turkish, Russian, and Hindi],
               use livekit.plugins.turn_detector.multilingual.MultilingualModel
            3. If the language is defined but not in the above list, return "vad"
        3. If no language can be determined, return NotGiven (let the AgentSession decide the best turn detection mode)
    Args:
        agent_config (_type_): _description_

    Returns:
        NotGivenOr[TurnDetectionMode]: _description_
    """
    try:
        stt_provider = agent_config["live_captions"]["provider"]
    except Exception:
        return NotGiven

    # Rule 1: AssemblyAI uses STT endpointing
    if stt_provider == "assemblyai":
        return "stt"

    # Rule 2: Determine language and select appropriate turn detector
    config_manager = ConfigManager(agent_config, f"live_captions.{stt_provider}")
    language = config_manager.configured_string_value("language")

    # If no language is configured, try to get the default from the STT plugin constructor
    if language is NOT_PROVIDED:
        language = _get_stt_language_default(stt_provider)

    # If still no language, return NotGiven
    if language is None:
        return NotGiven

    # Rule 2.1: If language starts with "en", use English model
    if isinstance(language, str) and language.lower().startswith("en"):
        from livekit.plugins.turn_detector.english import EnglishModel

        return EnglishModel()

    # Rule 2.2: For other supported languages, use Multilingual model
    if isinstance(language, str) and any(
        language.lower().startswith(prefix)
        for prefix in SUPPORTED_LANGUAGES_IN_MULTILINGUAL_TURN_DETECTION
    ):
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        return MultilingualModel()

    # Rule 2.3: If language is defined but not supported by the multilingual model, use "vad"
    if isinstance(language, str):
        return "vad"

    # Rule 3: Cannot determine language
    return NotGiven


def _get_stt_language_default(stt_provider: str) -> str | None:
    """Get the default language parameter from the STT plugin constructor.

    Args:
        stt_provider: The name of the STT provider (e.g., "aws", "azure", "openai")

    Returns:
        The default language value if it exists, None otherwise
    """
    if stt_provider not in STT_PROVIDERS:
        return None

    provider_config = STT_PROVIDERS[stt_provider]
    module_name = provider_config.plugin_module
    class_name = provider_config.plugin_class

    try:
        # Dynamically import the module
        import importlib

        module = importlib.import_module(module_name)
        stt_class = getattr(module, class_name)

        # Get the signature of the __init__ method
        sig = inspect.signature(stt_class.__init__)

        # Check if 'language' parameter exists and has a default
        if "language" in sig.parameters:
            default = sig.parameters["language"].default
            if default is not inspect.Parameter.empty:
                return default
    except Exception as e:
        logging.error(f"Could not get language default for {stt_provider}: {e}")

    return None


# Initialize the registry when the module is loaded
_initialize_stt_registry()
