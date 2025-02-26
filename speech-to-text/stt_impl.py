import logging
import os
from livekit.plugins import azure, google, openai, deepgram, assemblyai, fal, clova

logger = logging.getLogger("agent")


def get_azure_stt_impl(agent_config):
    wrong_azure_config_msg = "Wrong azure credentials. One of these combinations must be set:\n    - speech_host\n    - speech_key + speech_region\n   - speech_auth_token + speech_region"
    try:
        speech_host = agent_config["stt"]["azure"]["speech_host"]
    except TypeError or KeyError:
        speech_host = None
    if speech_host is not None:
        return azure.STT(speech_host=speech_host)
    else:
        try:
            speech_region = agent_config["stt"]["azure"]["speech_region"]
        except TypeError or KeyError:
            raise ValueError(wrong_azure_config_msg)
        if speech_region is None:
            raise ValueError(wrong_azure_config_msg)
        try:
            speech_key = agent_config["stt"]["azure"]["speech_key"]
        except TypeError or KeyError:
            speech_key = None
        if speech_key is not None:
            return azure.STT(speech_key=speech_key, speech_region=speech_region)
        else:
            try:
                speech_auth_token = agent_config["stt"]["azure"]["speech_auth_token"]
            except TypeError or KeyError:
                raise ValueError(wrong_azure_config_msg)
            if speech_auth_token is None:
                raise ValueError(wrong_azure_config_msg)
            else:
                return azure.STT(
                    speech_auth_token=speech_auth_token,
                    speech_region=speech_region,
                )


def get_google_stt_impl(agent_config):
    wrong_google_config_msg = "Wrong Google credentials. stt.google.credentials_file must be set"
    try:
        credentials_file = agent_config["stt"]["google"]["credentials_file"]
    except TypeError or KeyError:
        raise ValueError(wrong_google_config_msg)
    return google.STT(
        model="chirp_2",
        spoken_punctuation=True,
        credentials_file=credentials_file,
    )


def get_openai_stt_impl(agent_config):
    wrong_openai_config_msg = "Wrong OpenAI credentials. stt.openai.api_key must be set"
    try:
        api_key = agent_config["stt"]["openai"]["api_key"]
    except TypeError or KeyError:
        raise ValueError(wrong_openai_config_msg)
    if api_key is None:
        raise ValueError(wrong_openai_config_msg)
    return openai.STT(api_key=api_key)


def get_groq_stt_impl(agent_config):
    wrong_groq_config_msg = "Wrong Groq credentials. stt.groq.api_key must be set"
    default_model = "whisper-large-v3-turbo"
    default_language = "en"
    try:
        api_key = agent_config["stt"]["groq"]["api_key"]
    except TypeError or KeyError:
        raise ValueError(wrong_groq_config_msg)
    if api_key is None:
        raise ValueError(wrong_groq_config_msg)
    model = agent_config["stt"]["groq"]["model"]
    language = agent_config["stt"]["groq"]["language"]
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
        "Wrong Deepgram credentials. stt.deepgram.api_key must be set"
    )
    default_model = "nova-2-general"
    default_language = "en-US"
    try:
        api_key = agent_config["stt"]["deepgram"]["api_key"]
    except TypeError or KeyError:
        raise ValueError(wrong_deepgram_config_msg)
    if api_key is None:
        raise ValueError(wrong_deepgram_config_msg)
    model = agent_config["stt"]["deepgram"]["model"]
    language = agent_config["stt"]["deepgram"]["language"]
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
        "Wrong AssemblyAI credentials. stt.assemblyai.api_key must be set"
    )
    try:
        api_key = agent_config["stt"]["assemblyai"]["api_key"]
    except TypeError or KeyError:
        raise ValueError(wrong_assemblyai_config_msg)
    if api_key is None:
        raise ValueError(wrong_assemblyai_config_msg)
    return assemblyai.STT(api_key=api_key)


def get_fal_stt_impl(agent_config):
    wrong_fal_config_msg = "Wrong FAL credentials. stt.fal.api_key must be set"
    try:
        api_key = agent_config["stt"]["fal"]["api_key"]
    except TypeError or KeyError:
        raise ValueError(wrong_fal_config_msg)
    if api_key is None:
        raise ValueError(wrong_fal_config_msg)
    # livekit-plugins-fal require the FAL_KEY env var to be set
    os.environ["FAL_KEY"] = api_key
    return fal.WizperSTT(
        language=agent_config.get("stt.fal.language"),
    )


def get_clova_stt_impl(agent_config):
    wrong_clova_config_msg = (
        "Wrong Clova credentials. stt.clova.api_key and stt.clova.api_key must be set"
    )
    try:
        api_key = agent_config["stt"]["clova"]["api_key"]
        invoke_url = agent_config["stt"]["clova"]["invoke_url"]
    except TypeError or KeyError:
        raise ValueError(wrong_clova_config_msg)
    if invoke_url is None or api_key is None:
        raise ValueError(wrong_clova_config_msg)
    language = agent_config["stt"]["clova"]["language"]
    if language is None:
        return clova.STT(invoke_url=invoke_url, secret=api_key)
    else:
        return clova.STT(invoke_url=invoke_url, secret=api_key, language=language)


def get_stt_impl(agent_config):
    try:
        stt_provider = agent_config["stt"]["provider"]
    except TypeError or KeyError:
        stt_provider = None
    if stt_provider is None:
        raise ValueError("stt.provider not defined in agent configuration")
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
    else:
        raise ValueError(f"unknown STT provider: {stt_provider}")
