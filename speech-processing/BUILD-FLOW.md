# Build Flow Diagram

This document illustrates the complete build and deployment flow for the Nuitka-based binarization.

## Standard vs Binary Build Flow

### Standard Build Flow (Dockerfile.sherpa)

```
┌─────────────────────────────────────────┐
│  Developer Machine                       │
│  ┌────────────────────────────────┐    │
│  │ 1. Source Code (.py files)     │    │
│  │    - main.py                    │    │
│  │    - stt_impl.py               │    │
│  │    - vad_stt_wrapper.py        │    │
│  └────────────────┬───────────────┘    │
│                   │                      │
│                   ▼                      │
│  ┌────────────────────────────────┐    │
│  │ 2. Docker Build                 │    │
│  │    docker build -f Dockerfile.  │    │
│  │    sherpa ...                   │    │
│  └────────────────┬───────────────┘    │
│                   │                      │
└───────────────────┼──────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Docker Image (Standard)                 │
│  ┌────────────────────────────────┐    │
│  │ /app/main.py          ◄── READABLE  │
│  │ /app/stt_impl.py      ◄── READABLE  │
│  │ /app/vad_stt_wrapper.py ◄── READABLE│
│  │ /opt/venv/...         ◄── Python    │
│  │ /app/sherpa-onnx-streaming-models   │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    │
                    ▼
                 DEPLOY
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Runtime Container                       │
│  python main.py start  ◄── Interpreted  │
└─────────────────────────────────────────┘
```

### Binary Build Flow (Dockerfile.sherpa-binary)

```
┌─────────────────────────────────────────┐
│  Developer Machine                       │
│  ┌────────────────────────────────┐    │
│  │ 1. Prepare                      │    │
│  │    ./download-models.sh         │    │
│  │    export GITHUB_TOKEN=...      │    │
│  └────────────────┬───────────────┘    │
│                   │                      │
│                   ▼                      │
│  ┌────────────────────────────────┐    │
│  │ 2. Validate                     │    │
│  │    ./validate-binary-setup.sh   │    │
│  └────────────────┬───────────────┘    │
│                   │                      │
│                   ▼                      │
│  ┌────────────────────────────────┐    │
│  │ 3. Build Binary Image           │    │
│  │    ./build-sherpa-binary.sh     │    │
│  └────────────────┬───────────────┘    │
│                   │                      │
└───────────────────┼──────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Multi-Stage Build   │
        └─────────┬───────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│Stage 1 │  │ Stage 2  │  │ Stage 3  │
│Builder │─▶│Compiler  │─▶│ Runtime  │
└────────┘  └──────────┘  └──────────┘
    │             │             │
    ▼             ▼             ▼
Install      Compile       Package
Dependencies  with          Binary
+ Nuitka      Nuitka        Only
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Docker Image (Binary)                   │
│  ┌────────────────────────────────┐    │
│  │ /app/agent_dist/main.bin  ◄── BINARY   │
│  │      (ELF executable)              │
│  │                                     │
│  │ /app/sherpa-onnx-streaming-models  │
│  │ /app/.cache/huggingface            │
│  │                                     │
│  │ NO .py FILES! ✓                    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    │
                    ▼
                 DEPLOY
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Runtime Container                       │
│  ./agent_dist/main.bin start  ◄── Native    │
└─────────────────────────────────────────┘
```

## Detailed Build Stages

### Stage 1: Builder (Dependency Installation)

```
┌──────────────────────────────────────────────┐
│ FROM python:3.12-slim                         │
├──────────────────────────────────────────────┤
│ Install System Packages:                      │
│  • gcc, g++ (C/C++ compilers)                │
│  • python3-dev (Python headers)              │
│  • git (for GitHub repos)                    │
│  • ccache (compilation cache)                │
│  • patchelf (binary patching)                │
├──────────────────────────────────────────────┤
│ Create Virtual Environment:                   │
│  python -m venv /opt/venv                    │
├──────────────────────────────────────────────┤
│ Install Python Dependencies:                  │
│  • livekit-agents[silero]                    │
│  • livekit-plugins-sherpa (private repo)     │
│  • openviduagentutils (private repo)         │
│  • nuitka==2.5.6                            │
│  • ordered-set, zstandard                    │
├──────────────────────────────────────────────┤
│ ⚠ GitHub Token Used Here (then discarded)    │
└──────────────────────────────────────────────┘
```

### Stage 2: Compiler (Nuitka Compilation)

```
┌──────────────────────────────────────────────┐
│ FROM builder AS compiler                      │
├──────────────────────────────────────────────┤
│ Copy Source Files:                            │
│  • main.py                                   │
│  • stt_impl.py                               │
│  • vad_stt_wrapper.py                        │
├──────────────────────────────────────────────┤
│ Copy Models:                                  │
│  • sherpa-onnx-streaming-models/            │
├──────────────────────────────────────────────┤
│ Pre-download Dependencies:                    │
│  python main.py download-files               │
│  • Silero VAD models                         │
│  • HuggingFace cache                         │
├──────────────────────────────────────────────┤
│ Nuitka Compilation:                           │
│  python -m nuitka \                          │
│    --standalone \                            │
│    --follow-imports \                        │
│    --include-package=livekit \               │
│    --include-package=sherpa_onnx \           │
│    --nofollow-import-to=livekit.plugins.aws \│
│    ... (exclude unused plugins) ...          │
│    main.py                                   │
├──────────────────────────────────────────────┤
│ Output:                                       │
│  /build/dist/main.dist/                      │
│    ├── main (executable)                     │
│    ├── libpython3.12.so                      │
│    └── ... (dependencies)                    │
└──────────────────────────────────────────────┘
```

### Stage 3: Runtime (Final Image)

```
┌──────────────────────────────────────────────┐
│ FROM python:3.12-slim                         │
├──────────────────────────────────────────────┤
│ Install Runtime Dependencies:                 │
│  • libglib2.0-0 (minimal libraries)          │
├──────────────────────────────────────────────┤
│ Copy From Compiler Stage:                     │
│  • /build/dist/main.dist → /app/agent_dist  │
│  • /build/sherpa-onnx-streaming-models       │
│  • /build/.cache/huggingface                 │
├──────────────────────────────────────────────┤
│ Security Configuration:                       │
│  • Create non-privileged user (appuser)      │
│  • Set file ownership                        │
│  • Switch to non-root user                   │
├──────────────────────────────────────────────┤
│ Runtime Configuration:                        │
│  • HF_HUB_OFFLINE=1 (offline mode)           │
│  • PYTHONUNBUFFERED=1                        │
├──────────────────────────────────────────────┤
│ Entrypoint:                                   │
│  CMD ["./agent_dist/main.bin", "start"]          │
├──────────────────────────────────────────────┤
│ ✓ NO Python source files                     │
│ ✓ NO GitHub token                            │
│ ✓ Minimal attack surface                     │
└──────────────────────────────────────────────┘
```

## Security Token Flow

```
┌──────────────────────────────────────────────┐
│  GitHub Token Handling                        │
└──────────────────────────────────────────────┘

Developer Machine:
  export GITHUB_TOKEN=ghp_xxxxx
         │
         ▼
Docker Build (BuildKit Secrets):
  --secret id=github_token,env=GITHUB_TOKEN
         │
         ▼
Stage 1 (Builder):
  ┌─────────────────────────────────┐
  │ RUN --mount=type=secret,...     │
  │   GITHUB_TOKEN=$(cat /run/...) │
  │   git config credential.helper  │
  │   pip install -r requirements   │
  │   rm ~/.git-credentials         │
  └─────────────────────────────────┘
         │
         ▼ Token used
         │
         ▼ Credentials deleted
         │
         ▼ Stage discarded
         
Stage 2 (Compiler):
  ✓ Token not accessible
  
Stage 3 (Runtime):
  ✓ Token not accessible
  ✓ No GitHub token in final image
  ✓ No token in image layers
  ✓ No token in build cache
```

## Module Inclusion Strategy

```
┌──────────────────────────────────────────────┐
│  Nuitka Module Inclusion                     │
└──────────────────────────────────────────────┘

Application Code:
  ✅ --include-module=stt_impl
  ✅ --include-module=vad_stt_wrapper
  ✅ main.py (entry point)

LiveKit Framework:
  ✅ --include-package=livekit
  ✅ --include-package=livekit.agents
  ✅ --include-package=livekit.rtc

Required Plugins:
  ✅ --include-package=livekit.plugins.sherpa
  ✅ --include-package=livekit.plugins.silero
  ✅ --include-package=sherpa_onnx

Custom Utilities:
  ✅ --include-package=openviduagentutils

Python Stdlib:
  ✅ --include-package=asyncio
  ✅ (automatic via --follow-imports)

EXCLUDED (Not Needed for Sherpa):
  ❌ --nofollow-import-to=livekit.plugins.aws
  ❌ --nofollow-import-to=livekit.plugins.azure
  ❌ --nofollow-import-to=livekit.plugins.openai
  ❌ --nofollow-import-to=livekit.plugins.google
  ❌ --nofollow-import-to=livekit.plugins.groq
  ❌ --nofollow-import-to=livekit.plugins.deepgram
  ❌ ... (10+ other cloud providers)
  ❌ --nofollow-import-to=livekit.plugins.vosk

Result:
  • Smaller binary size
  • Faster compilation
  • Only necessary code included
```

## File System Comparison

### Standard Deployment

```
Container: openvidu/agent-speech-processing-sherpa:main

/app/
├── main.py                          📄 READABLE
├── stt_impl.py                      📄 READABLE
├── vad_stt_wrapper.py               📄 READABLE
├── sherpa-onnx-streaming-models/
│   ├── encoder-epoch-99.onnx
│   ├── decoder-epoch-99.onnx
│   └── ...
└── .cache/
    └── huggingface/

/opt/venv/
├── lib/python3.12/site-packages/
│   ├── livekit/                     📄 READABLE
│   ├── openviduagentutils/          📄 READABLE
│   └── ...
└── bin/
    └── python

Total: ~1.5-2 GB
Source Protection: ❌ NONE
```

### Binary Deployment

```
Container: openvidu/agent-speech-processing-sherpa-binary:main

/app/
├── agent_dist/
│   ├── main                         🔒 BINARY
│   ├── libpython3.12.so.1.0        🔒 BINARY
│   ├── _multiprocessing.cpython-312-x86_64-linux-gnu.so
│   ├── ... (compiled dependencies)
│   └── ... (NO .py files!)
├── sherpa-onnx-streaming-models/
│   ├── encoder-epoch-99.onnx
│   ├── decoder-epoch-99.onnx
│   └── ...
└── .cache/
    └── huggingface/

Total: ~2-3 GB
Source Protection: ✅ STRONG
```

## Verification Commands

```bash
# Standard Image
$ docker run --rm -it \
    --entrypoint /bin/bash \
    openvidu/agent-speech-processing-sherpa:main

appuser@container:/app$ cat main.py
#!/usr/bin/env python3
import asyncio
...  # ← Full source code visible! ❌

# Binary Image
$ docker run --rm -it \
    --entrypoint /bin/bash \
    openvidu/agent-speech-processing-sherpa-binary:main

appuser@container:/app$ cat main.py
cat: main.py: No such file or directory  # ✅

appuser@container:/app$ cat agent_dist/main.bin
�ELF�^@�^@^A^@^@^@^@^@^@^@...  # ← Binary! ✅

appuser@container:/app$ file agent_dist/main.bin
agent_dist/main.bin: ELF 64-bit LSB executable, x86-64  # ✅

appuser@container:/app$ strings agent_dist/main.bin | grep "import"
# Only see compiled strings, not Python code ✅
```

## Performance Comparison

```
┌─────────────────┬──────────────┬───────────────┐
│     Metric      │   Standard   │    Binary     │
├─────────────────┼──────────────┼───────────────┤
│ Build Time      │   2-5 min    │  10-20 min    │
│ Image Size      │  1.5-2 GB    │   2-3 GB      │
│ Startup Time    │   Normal     │ Slightly faster│
│ Runtime Speed   │   Baseline   │   Similar     │
│ Memory Usage    │   Baseline   │   Similar     │
│ Code Protection │     None     │    Strong     │
└─────────────────┴──────────────┴───────────────┘
```

## Summary

The binary build flow provides:

✅ **Strong Code Protection** - No readable Python source  
✅ **Production Ready** - Multi-stage build, non-root user  
✅ **Secure** - GitHub token isolated, no secrets in image  
✅ **Optimized** - Unused plugins excluded, compiled code  
✅ **Maintainable** - Clear build process, good documentation  

Trade-offs:

⚠️ **Longer Build Time** - Nuitka compilation takes 10-20 minutes  
⚠️ **Larger Image** - Binary + embedded Python adds ~1 GB  
⚠️ **Harder Debugging** - Binary stack traces, no source context  

**Recommendation:** Use binary deployment for customer-facing or SaaS deployments where code protection is critical. Use standard deployment for internal tools and development.
