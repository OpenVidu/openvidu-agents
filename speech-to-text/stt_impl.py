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
)
from livekit.plugins.speechmatics.types import TranscriptionConfig
from config_utils import ConfigManager

logger = logging.getLogger("agent")


def get_aws_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.aws")
    wrong_credentials = "Wrong AWS credentials. speech_to_text.aws.aws_access_key_id, speech_to_text.aws.aws_secret_access_key and speech_to_text.aws.aws_default_region must be set"

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

    language = config_manager.optional_value("language", "en-US")
    vocabulary_name = config_manager.optional_value("vocabulary_name", None)
    language_model_name = config_manager.optional_value("language_model_name", None)
    enable_partial_results_stabilization = config_manager.optional_value(
        "enable_partial_results_stabilization", None
    )
    partial_results_stability = config_manager.optional_value(
        "partial_results_stability", None
    )
    vocab_filter_name = config_manager.optional_value("vocab_filter_name", None)
    vocab_filter_method = config_manager.optional_value("vocab_filter_method", None)

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


def get_azure_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.azure")
    wrong_credentials = "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n    - speech_auth_token + speech_region"

    speech_host = config_manager.optional_value("speech_host", None)
    speech_region = config_manager.optional_value("speech_region", None)
    speech_key = config_manager.optional_value("speech_key", None)
    speech_auth_token = config_manager.optional_value("speech_auth_token", None)
    languages = config_manager.optional_value("languages", ["en-US"])

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
        )
    else:
        raise ValueError(wrong_credentials)


def get_google_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.google")
    wrong_credentials = (
        "Wrong Google credentials. speech_to_text.google.credentials_info must be set"
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
    model = config_manager.optional_value("model", "latest_long")
    languages = config_manager.optional_value("languages", "en-US")
    detect_language = config_manager.optional_value("detect_language", True)
    location = config_manager.optional_value("location", "us-central1")
    punctuate = config_manager.optional_value("punctuate", True)
    spoken_punctuation = config_manager.optional_value("spoken_punctuation", False)
    interim_results = config_manager.optional_value("interim_results", True)

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


def get_openai_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.openai")
    wrong_credentials = (
        "Wrong OpenAI credentials. speech_to_text.openai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    return openai.STT(api_key=api_key)


def get_groq_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.groq")
    wrong_credentials = (
        "Wrong Groq credentials. speech_to_text.groq.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.optional_value("model", "whisper-large-v3-turbo")
    language = config_manager.optional_value("language", "en")

    return openai.stt.STT.with_groq(
        api_key=api_key,
        model=model,
        language=language,
    )


def get_deepgram_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.deepgram")
    wrong_credentials = (
        "Wrong Deepgram credentials. speech_to_text.deepgram.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    model = config_manager.optional_value("model", "nova-2-general")
    language = config_manager.optional_value("language", "en-US")
    interim_results = config_manager.optional_value("interim_results", True)
    smart_format = config_manager.optional_value("smart_format", True)
    punctuate = config_manager.optional_value("punctuate", True)
    filler_words = config_manager.optional_value("filler_words", True)
    profanity_filter = config_manager.optional_value("profanity_filter", False)
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


def get_assemblyai_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.assemblyai")
    wrong_credentials = (
        "Wrong AssemblyAI credentials. speech_to_text.assemblyai.api_key must be set"
    )

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)

    return assemblyai.STT(api_key=api_key)


def get_fal_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.fal")
    wrong_credentials = "Wrong FAL credentials. speech_to_text.fal.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_value("language", "en")

    # livekit-plugins-fal require the FAL_KEY env var to be set
    os.environ["FAL_KEY"] = api_key

    return fal.WizperSTT(language=language)


def get_clova_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.clova")
    wrong_credentials = "Wrong Clova credentials. speech_to_text.clova.api_key and speech_to_text.clova.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    invoke_url = config_manager.mandatory_value("invoke_url", wrong_credentials)
    language = config_manager.optional_value("language", "en-US")

    return clova.STT(invoke_url=invoke_url, secret=api_key, language=language)


def get_speechmatics_stt_impl(agent_config):

    config_manager = ConfigManager(agent_config, "speech_to_text.speechmatics")
    wrong_credentials = "Wrong Speechmatics credentials. speech_to_text.speechmatics.api_key must be set"

    api_key = config_manager.mandatory_value("api_key", wrong_credentials)
    language = config_manager.optional_value("language", "en")

    # livekit-plugins-speechmatics require the SPEECHMATICS_API_KEY env var to be set
    os.environ["SPEECHMATICS_API_KEY"] = api_key

    transcrpition_config = TranscriptionConfig(
        language=language,
        # Default values of livekit-plugin
        operating_point="enhanced",
        enable_partials=True,
        max_delay=0.7,
    )
    return speechmatics.STT(transcription_config=transcrpition_config)


def get_stt_impl(agent_config):
    try:
        stt_provider = agent_config["speech_to_text"]["provider"]
    except Exception:
        stt_provider = None
    if stt_provider is None:
        raise ValueError("speech_to_text.provider not defined in agent configuration")
    else:
        print("Using", stt_provider, "as STT provider")
    if stt_provider == "aws":
        return get_aws_stt_impl(agent_config)
    if stt_provider == "azure":
        return get_azure_stt_impl(agent_config)
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
    else:
        raise ValueError(f"unknown STT provider: {stt_provider}")
