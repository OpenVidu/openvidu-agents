import logging
import os
import json
import tempfile
from livekit.plugins import (
    aws,
    azure,
    google,
    openai,
    deepgram,
    assemblyai,
    fal,
    clova,
    speechmatics,
    gladia,
    groq,
    sarvam,
)
from livekit.plugins.speechmatics.types import TranscriptionConfig
from livekit.agents import stt
from openviduagentutils.config_manager import ConfigManager

from azure.cognitiveservices.speech.enums import ProfanityOption


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

    language = config_manager.optional_string_value("language", "en-US")
    vocabulary_name = config_manager.optional_string_value("vocabulary_name", None)
    language_model_name = config_manager.optional_string_value(
        "language_model_name", None
    )
    enable_partial_results_stabilization = config_manager.optional_boolean_value(
        "enable_partial_results_stabilization", None
    )
    partial_results_stability = config_manager.optional_string_value(
        "partial_results_stability", None
    )
    vocab_filter_name = config_manager.optional_string_value("vocab_filter_name", None)
    vocab_filter_method = config_manager.optional_string_value(
        "vocab_filter_method", None
    )

    return aws.STT(
        api_key=api_key,
        api_secret=api_secret,
        speech_region=default_region,
        language=language,
        vocabulary_name=vocabulary_name,
        language_model_name=language_model_name,
        enable_partial_results_stabilization=enable_partial_results_stabilization,
        partial_results_stability=partial_results_stability,
        vocab_filter_name=vocab_filter_name,
        vocab_filter_method=vocab_filter_method,
    )


def get_azure_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.azure")
    wrong_credentials = "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n    - speech_auth_token + speech_region"

    speech_host = config_manager.optional_string_value("speech_host", None)
    speech_region = config_manager.optional_string_value("speech_region", None)
    speech_key = config_manager.optional_string_value("speech_key", None)
    speech_auth_token = config_manager.optional_string_value("speech_auth_token", None)
    languages = config_manager.optional_value("languages", ["en-US"])
    profanity = config_manager.optional_enum_value("profanity", ProfanityOption, None)

    if speech_host is not None:
        return azure.STT(speech_host=speech_host, languages=languages)
    elif speech_key is not None and speech_region is not None:
        return azure.STT(
            speech_key=speech_key, speech_region=speech_region, languages=languages
        )
    elif speech_auth_token is not None and speech_region is not None:
        return azure.STT(
            speech_auth_token=speech_auth_token,
            speech_region=speech_region,
            languages=languages,
            profanity=profanity,
        )
    else:
        raise ValueError(wrong_credentials)


def get_azure_openai_stt_impl(agent_config) -> stt.STT:
    config_manager = ConfigManager(agent_config, "live_captions.azure_openai")

    azure_api_key = config_manager.mandatory_value(
        "azure_api_key",
        "Wrong Azure OpenAI credentials. live_captions.azure_openai.azure_api_key must be set",
    )
    azure_ad_token = config_manager.mandatory_value(
        "azure_ad_token",
        "Wrong Azure OpenAI credentials. live_captions.azure_openai.azure_ad_token must be set",
    )
    azure_endpoint = config_manager.mandatory_value(
        "azure_endpoint",
        "Wrong Azure OpenAI configuration. live_captions.azure_openai.azure_endpoint must be set",
    )
    api_version = config_manager.mandatory_value(
        "api_version",
        "Wrong Azure OpenAI configuration. live_captions.azure_openai.api_version must be set",
    )
    azure_deployment = config_manager.optional_string_value("azure_deployment", None)
    organization = config_manager.optional_string_value("organization", None)
    project = config_manager.optional_string_value("project", None)
    language = config_manager.optional_string_value("language", "en")
    model = config_manager.optional_string_value("model", "gpt-4o-mini-transcribe")
    prompt = config_manager.optional_string_value("prompt", None)

    return openai.STT.with_azure(
        api_key=azure_api_key,
        azure_ad_token=azure_ad_token,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=azure_deployment,
        organization=organization,
        project=project,
        language=language,
        model=model,
        prompt=prompt,
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
    model = config_manager.optional_string_value("model", "latest_long")
    languages = config_manager.optional_string_value("languages", "en-US")
    detect_language = config_manager.optional_boolean_value("detect_language", True)
    location = config_manager.optional_string_value("location", "us-central1")
    punctuate = config_manager.optional_boolean_value("punctuate", True)
    spoken_punctuation = config_manager.optional_boolean_value(
        "spoken_punctuation", False
    )
    interim_results = config_manager.optional_boolean_value("interim_results", True)

    return google.STT(
        credentials_file=temp_file.name,
        model=model,
        languages=languages,
        detect_language=detect_language,
        location=location,
        punctuate=punctuate,
        spoken_punctuation=spoken_punctuation,
        interim_results=interim_results,
    )


def get_openai_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.openai")
    wrong_credentials = (
        "Wrong OpenAI credentials. live_captions.openai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.optional_string_value("model", "whisper-1")
    language = config_manager.optional_string_value("language", "en")

    return openai.STT(api_key=api_key, model=model, language=language)


def get_groq_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "live_captions.groq")
    wrong_credentials = "Wrong Groq credentials. live_captions.groq.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.optional_string_value("model", "whisper-large-v3-turbo")
    language = config_manager.optional_string_value("language", "en")
    prompt = config_manager.optional_string_value("prompt", None)

    return groq.STT(
        api_key=api_key,
        model=model,
        language=language,
        prompt=prompt,
    )


def get_deepgram_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.deepgram")
    wrong_credentials = (
        "Wrong Deepgram credentials. live_captions.deepgram.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.optional_string_value("model", "nova-2-general")
    language = config_manager.optional_string_value("language", "en-US")
    interim_results = config_manager.optional_boolean_value("interim_results", True)
    smart_format = config_manager.optional_boolean_value("smart_format", True)
    punctuate = config_manager.optional_boolean_value("punctuate", True)
    filler_words = config_manager.optional_boolean_value("filler_words", True)
    profanity_filter = config_manager.optional_boolean_value("profanity_filter", False)
    keywords = config_manager.optional_value("keywords", None)
    keyterms = config_manager.optional_value("keyterms", None)

    return deepgram.stt.STT(
        api_key=api_key,
        model=model,
        language=language,
        interim_results=interim_results,
        smart_format=smart_format,
        punctuate=punctuate,
        filler_words=filler_words,
        profanity_filter=profanity_filter,
        keywords=keywords,
        keyterms=keyterms,
    )


def get_assemblyai_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.assemblyai")
    wrong_credentials = (
        "Wrong AssemblyAI credentials. live_captions.assemblyai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    format_turns = config_manager.optional_boolean_value("format_turns", None)

    return assemblyai.STT(api_key=api_key, format_turns=format_turns)


def get_fal_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.fal")
    wrong_credentials = "Wrong FAL credentials. live_captions.fal.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "en")

    return fal.WizperSTT(api_key=api_key, language=language)


def get_clova_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.clova")
    wrong_credentials = "Wrong Clova credentials. live_captions.clova.api_key and live_captions.clova.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    invoke_url = config_manager.mandatory_value("invoke_url", wrong_credentials)
    language = config_manager.optional_string_value("language", "en-US")
    threshold = config_manager.optional_numeric_value("threshold", 0.5)

    return clova.STT(
        invoke_url=invoke_url, secret=api_key, language=language, threshold=threshold
    )


def get_speechmatics_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.speechmatics")
    wrong_credentials = (
        "Wrong Speechmatics credentials. live_captions.speechmatics.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "en")
    operating_point = config_manager.optional_string_value(
        "operating_point", "enhanced"
    )
    enable_partials = config_manager.optional_boolean_value("enable_partials", True)
    output_locale = config_manager.optional_string_value("output_locale", None)
    max_delay = config_manager.optional_numeric_value("max_delay", 0.7)
    max_delay_mode = config_manager.optional_string_value("max_delay_mode", "flexible")
    punctuation_overrides = config_manager.optional_dict_value(
        "punctuation_overrides", None
    )
    additional_vocab = config_manager.optional_value("additional_vocab", None)
    speaker_diarization_config = config_manager.optional_dict_value(
        "speaker_diarization_config", None
    )

    # livekit-plugins-speechmatics require the SPEECHMATICS_API_KEY env var to be set
    os.environ["SPEECHMATICS_API_KEY"] = api_key

    transcrpition_config = TranscriptionConfig(
        operating_point=operating_point,
        language=language,
        output_locale=output_locale,
        punctuation_overrides=punctuation_overrides,
        additional_vocab=additional_vocab,
        enable_partials=enable_partials,
        max_delay=max_delay,
        max_delay_mode=max_delay_mode,
        speaker_diarization_config=speaker_diarization_config,
    )
    return speechmatics.STT(transcription_config=transcrpition_config)


def get_gladia_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.gladia")
    wrong_credentials = (
        "Wrong Gladia credentials. live_captions.gladia.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    interim_results = config_manager.optional_boolean_value("interim_results", True)
    languages = config_manager.optional_value("languages", None)
    code_switching = config_manager.optional_boolean_value("code_switching", True)

    return gladia.STT(
        api_key=api_key,
        interim_results=interim_results,
        languages=languages,
        code_switching=code_switching,
    )


def get_sarvam_stt_impl(agent_config) -> stt.STT:

    config_manager = ConfigManager(agent_config, "live_captions.sarvam")
    wrong_credentials = (
        "Wrong Sarvam credentials. live_captions.sarvam.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_string_value("language", "unknown")
    model = config_manager.optional_string_value("model", None)

    return sarvam.STT(
        api_key=api_key,
        language=language,
        model=model,
    )


def get_stt_impl(agent_config) -> stt.STT:
    try:
        stt_provider = agent_config["live_captions"]["provider"]
    except Exception:
        stt_provider = None
    if stt_provider is None:
        raise ValueError("live_captions.provider not defined in agent configuration")
    else:
        logging.info(f"Using {stt_provider} as STT provider")
    if stt_provider == "aws":
        return get_aws_stt_impl(agent_config)
    if stt_provider == "azure":
        return get_azure_stt_impl(agent_config)
    elif stt_provider == "azure_openai":
        return get_azure_openai_stt_impl(agent_config)
    elif stt_provider == "google":
        return get_google_stt_impl(agent_config)
    elif stt_provider == "openai":
        return get_openai_stt_impl(agent_config)
    elif stt_provider == "groq":
        return get_groq_stt_impl(agent_config)
    elif stt_provider == "deepgram":
        return get_deepgram_stt_impl(agent_config)
    elif stt_provider == "assemblyai":
        return get_assemblyai_stt_impl(agent_config)
    elif stt_provider == "fal":
        return get_fal_stt_impl(agent_config)
    elif stt_provider == "clova":
        return get_clova_stt_impl(agent_config)
    elif stt_provider == "speechmatics":
        return get_speechmatics_stt_impl(agent_config)
    elif stt_provider == "gladia":
        return get_gladia_stt_impl(agent_config)
    elif stt_provider == "sarvam":
        return get_sarvam_stt_impl(agent_config)
    else:
        raise ValueError(f"unknown STT provider: {stt_provider}")
