import logging
import os
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

logger = logging.getLogger("agent")


def get_aws_stt_impl(agent_config):
    wrong_aws_config_msg = "Wrong AWS credentials. speech_to_text.aws.aws_access_key_id, speech_to_text.aws.aws_secret_access_key and speech_to_text.aws.aws_default_region must be set"

    # Optional values: [language]
    try:
        language = agent_config["speech_to_text"]["aws"]["language"]
    except Exception:
        language = None

    try:
        api_key = agent_config["speech_to_text"]["aws"]["aws_access_key_id"]
        api_secret = agent_config["speech_to_text"]["aws"]["aws_secret_access_key"]
        default_region = agent_config["speech_to_text"]["aws"]["aws_default_region"]

        # FIX for: https://github.com/awslabs/amazon-transcribe-streaming-sdk/issues/7#issuecomment-1677230478
        os.environ["AWS_ACCESS_KEY_ID"] = api_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = api_secret
        os.environ["AWS_DEFAULT_REGION"] = default_region
    except Exception:
        raise ValueError(wrong_aws_config_msg)
    if api_key is None or api_secret is None or default_region is None:
        raise ValueError(wrong_aws_config_msg)
    if language is None:
        return aws.STT(
            api_key=api_key, api_secret=api_secret, speech_region=default_region
        )
    else:
        return aws.STT(
            api_key=api_key,
            api_secret=api_secret,
            speech_region=default_region,
            language=language,
        )


def get_azure_stt_impl(agent_config):
    wrong_azure_config_msg = "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n    - speech_auth_token + speech_region"

    # Optional values: [languages]
    try:
        languages = agent_config["speech_to_text"]["azure"]["languages"]
    except Exception:
        languages = None

    try:
        speech_host = agent_config["speech_to_text"]["azure"]["speech_host"]
    except Exception:
        speech_host = None
    if speech_host is not None:
        if languages is None:
            return azure.STT(speech_host=speech_host)
        else:
            return azure.STT(speech_host=speech_host, languages=languages)
    else:
        try:
            speech_region = agent_config["speech_to_text"]["azure"]["speech_region"]
        except Exception:
            raise ValueError(wrong_azure_config_msg)
        if speech_region is None:
            raise ValueError(wrong_azure_config_msg)
        try:
            speech_key = agent_config["speech_to_text"]["azure"]["speech_key"]
        except Exception:
            speech_key = None
        if speech_key is not None:
            if languages is None:
                return azure.STT(speech_key=speech_key, speech_region=speech_region)
            else:
                return azure.STT(
                    speech_key=speech_key,
                    speech_region=speech_region,
                    languages=languages,
                )
        else:
            try:
                speech_auth_token = agent_config["speech_to_text"]["azure"][
                    "speech_auth_token"
                ]
            except Exception:
                raise ValueError(wrong_azure_config_msg)
            if speech_auth_token is None:
                raise ValueError(wrong_azure_config_msg)
            else:
                if languages is None:
                    return azure.STT(
                        speech_auth_token=speech_auth_token,
                        speech_region=speech_region,
                    )
                else:
                    return azure.STT(
                        speech_auth_token=speech_auth_token,
                        speech_region=speech_region,
                        languages=languages,
                    )


def get_google_stt_impl(agent_config):
    wrong_google_config_msg = (
        "Wrong Google credentials. speech_to_text.google.credentials_file must be set"
    )
    try:
        credentials_file = agent_config["speech_to_text"]["google"]["credentials_file"]
    except Exception:
        raise ValueError(wrong_google_config_msg)
    return google.STT(
        model="chirp_2",
        spoken_punctuation=True,
        credentials_file=credentials_file,
    )


def get_openai_stt_impl(agent_config):
    wrong_openai_config_msg = (
        "Wrong OpenAI credentials. speech_to_text.openai.api_key must be set"
    )
    try:
        api_key = agent_config["speech_to_text"]["openai"]["api_key"]
    except Exception:
        raise ValueError(wrong_openai_config_msg)
    if api_key is None:
        raise ValueError(wrong_openai_config_msg)
    return openai.STT(api_key=api_key)


def get_groq_stt_impl(agent_config):
    wrong_groq_config_msg = (
        "Wrong Groq credentials. speech_to_text.groq.api_key must be set"
    )
    default_model = "whisper-large-v3-turbo"
    default_language = "en"
    try:
        api_key = agent_config["speech_to_text"]["groq"]["api_key"]
    except Exception:
        raise ValueError(wrong_groq_config_msg)
    if api_key is None:
        raise ValueError(wrong_groq_config_msg)
    try:
        model = agent_config["speech_to_text"]["groq"]["model"]
    except Exception:
        model = None
    try:
        language = agent_config["speech_to_text"]["groq"]["language"]
    except Exception:
        language = None
    if model is None:
        model = default_model
    if language is None:
        language = default_language
    return openai.stt.STT.with_groq(
        api_key=api_key,
        model=model,
        language=language,
    )


def get_deepgram_stt_impl(agent_config):
    wrong_deepgram_config_msg = (
        "Wrong Deepgram credentials. speech_to_text.deepgram.api_key must be set"
    )
    default_model = "nova-2-general"
    default_language = "en-US"
    try:
        api_key = agent_config["speech_to_text"]["deepgram"]["api_key"]
    except Exception:
        raise ValueError(wrong_deepgram_config_msg)
    if api_key is None:
        raise ValueError(wrong_deepgram_config_msg)
    model = agent_config["speech_to_text"]["deepgram"]["model"]
    language = agent_config["speech_to_text"]["deepgram"]["language"]
    if model is None:
        model = default_model
    if language is None:
        language = default_language
    return deepgram.stt.STT(
        model=model,
        language=language,
        interim_results=True,
        smart_format=True,
        punctuate=True,
        filler_words=True,
        profanity_filter=False,
        keywords=[("LiveKit", 1.5)],
    )


def get_assemblyai_stt_impl(agent_config):
    wrong_assemblyai_config_msg = (
        "Wrong AssemblyAI credentials. speech_to_text.assemblyai.api_key must be set"
    )
    try:
        api_key = agent_config["speech_to_text"]["assemblyai"]["api_key"]
    except Exception:
        raise ValueError(wrong_assemblyai_config_msg)
    if api_key is None:
        raise ValueError(wrong_assemblyai_config_msg)
    return assemblyai.STT(api_key=api_key)


def get_fal_stt_impl(agent_config):
    wrong_fal_config_msg = (
        "Wrong FAL credentials. speech_to_text.fal.api_key must be set"
    )
    try:
        api_key = agent_config["speech_to_text"]["fal"]["api_key"]
    except Exception:
        raise ValueError(wrong_fal_config_msg)
    if api_key is None:
        raise ValueError(wrong_fal_config_msg)
    # livekit-plugins-fal require the FAL_KEY env var to be set
    os.environ["FAL_KEY"] = api_key
    return fal.WizperSTT(
        language=agent_config["speech_to_text"]["fal"]["language"],
    )


def get_clova_stt_impl(agent_config):
    wrong_clova_config_msg = "Wrong Clova credentials. speech_to_text.clova.api_key and speech_to_text.clova.api_key must be set"
    try:
        api_key = agent_config["speech_to_text"]["clova"]["api_key"]
        invoke_url = agent_config["speech_to_text"]["clova"]["invoke_url"]
    except Exception:
        raise ValueError(wrong_clova_config_msg)
    if invoke_url is None or api_key is None:
        raise ValueError(wrong_clova_config_msg)
    try:
        language = agent_config["speech_to_text"]["clova"]["language"]
    except Exception:
        language = None
    if language is None:
        return clova.STT(invoke_url=invoke_url, secret=api_key)
    else:
        return clova.STT(invoke_url=invoke_url, secret=api_key, language=language)


def get_speechmatics_stt_impl(agent_config):
    wrong_speechmatics_config_msg = "Wrong Speechmatics credentials. speech_to_text.speechmatics.api_key must be set"
    try:
        api_key = agent_config["speech_to_text"]["speechmatics"]["api_key"]
    except Exception:
        raise ValueError(wrong_speechmatics_config_msg)
    if api_key is None:
        raise ValueError(wrong_speechmatics_config_msg)
    # livekit-plugins-speechmatics require the SPEECHMATICS_API_KEY env var to be set
    os.environ["SPEECHMATICS_API_KEY"] = api_key
    try:
        language = agent_config["speech_to_text"]["speechmatics"]["language"]
    except Exception:
        language = None
    if language is None:
        return speechmatics.STT()
    else:
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
