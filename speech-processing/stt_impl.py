import logging
import os
import json
import tempfile
from livekit.plugins.speechmatics.types import TranscriptionConfig
from livekit.agents import stt

from openviduagentutils.config_manager import ConfigManager
from openviduagentutils.not_provided import NOT_PROVIDED

from azure.cognitiveservices.speech.enums import ProfanityOption


def get_aws_stt_impl(agent_config) -> stt.STT:
    from livekit.plugins import aws

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
    from livekit.plugins import azure

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
    from livekit.plugins import openai

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
    from livekit.plugins import google

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
    from livekit.plugins import openai

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
    from livekit.plugins import groq

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
    from livekit.plugins import deepgram

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
    from livekit.plugins import assemblyai

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
    from livekit.plugins import fal

    config_manager = ConfigManager(agent_config, "live_captions.fal")
    wrong_credentials = "Wrong FAL credentials. live_captions.fal.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.configured_string_value("language")

    kwargs = {k: v for k, v in {"language": language}.items() if v is not NOT_PROVIDED}

    return fal.WizperSTT(api_key=api_key, **kwargs)


def get_clova_stt_impl(agent_config) -> stt.STT:
    from livekit.plugins import clova

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
    from livekit.plugins import speechmatics

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
    from livekit.plugins import gladia

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
    from livekit.plugins import sarvam

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
    from livekit.plugins import spitch

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


def get_mistral_stt_impl(agent_config) -> stt.STT:
    from livekit.plugins import mistralai

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
    from livekit.plugins import cartesia

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
    elif stt_provider == "spitch":
        return get_spitch_stt_impl(agent_config)
    elif stt_provider == "mistralai":
        return get_mistral_stt_impl(agent_config)
    elif stt_provider == "cartesia":
        return get_cartesia_stt_impl(agent_config)
    else:
        raise ValueError(f"unknown STT provider: {stt_provider}")
