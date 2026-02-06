# Python Code Binarization with Nuitka

This document explains the approach used to binarize the speech-processing agent for enhanced security and code protection.

## Overview

The `Dockerfile.sherpa-binary` implements Python code binarization using **Nuitka**, a Python-to-C++ compiler that produces native executables. This approach makes it significantly harder to reverse-engineer the application code compared to distributing raw Python files.

## Why Nuitka?

After evaluating multiple Python binarization tools, Nuitka was selected for the following reasons:

### Comparison: Nuitka vs Cython

| Feature                  | Nuitka                          | Cython                        |
|--------------------------|----------------------------------|-------------------------------|
| **Output**               | Standalone native executable     | Compiled modules (.so/.pyd)   |
| **Obfuscation Level**    | Strong (by design)              | Moderate (not main purpose)   |
| **Docker Deployment**    | Easiest (single binary)         | Requires multi-stage build    |
| **Reverse Engineering**  | Harder (complete binary)        | Possible with moderate effort |
| **Async/AsyncIO Support**| Excellent native support        | Good but requires more setup  |
| **Performance**          | Fast overall                    | Fast with type hints          |
| **Code Changes Required**| None (works with vanilla Python)| Often requires type hints     |

### Key Advantages for This Project

1. **Better Code Protection**: Nuitka produces a true native binary that bundles the entire Python interpreter and all dependencies, making it significantly harder to extract the original source code.

2. **Async Application Support**: The speech-processing agent is built on the LiveKit Agents framework, which heavily uses `asyncio`. Nuitka has excellent support for async applications out of the box.

3. **Easier Docker Deployment**: The compiled binary is self-contained, resulting in a cleaner Docker image with fewer dependencies.

4. **No Code Modifications**: Nuitka works with standard Python code without requiring type annotations or other modifications.

## Build Process

The `Dockerfile.sherpa-binary` uses a multi-stage build process:

### Stage 1: Dependency Installation (builder)
- Installs all Python dependencies including the private `livekit-plugins-sherpa` package
- Uses GitHub token secret for authentication (isolated in builder stage)
- Installs Nuitka compiler and its dependencies
- **Security Note**: The GitHub token is only used in this stage and never appears in the final image

### Stage 2: Nuitka Compilation (compiler)
- Copies Python source files to a clean build directory
- Pre-downloads ML models and dependencies
- Compiles the agent using Nuitka with these key options:
  - `--standalone`: Creates a self-contained distribution
  - `--follow-imports`: Includes all imported modules
  - `--include-package`: Explicitly includes packages that might be dynamically imported
  - `--lto=yes`: Enables link-time optimization for better performance
  - `--jobs=auto`: Uses all CPU cores for faster compilation
- **Output**: Native binary executable in `/build/dist/main.dist/`

### Stage 3: Final Runtime Image
- Starts from a minimal Python slim image
- Copies only the compiled binary and required models
- No Python source code is present in this stage
- Runs as a non-privileged user for security
- **Result**: Secure container with binarized code

## Building the Docker Image

### Prerequisites

1. Download the Sherpa ONNX models:
   ```bash
   cd speech-processing
   ./download-models.sh
   ```

2. Create a GitHub Personal Access Token (PAT) with `repo` scope at:
   https://github.com/settings/tokens

### Build Commands

**Option 1 - Using environment variable:**
```bash
export GITHUB_TOKEN=ghp_your_token_here
docker build \
  --secret id=github_token,env=GITHUB_TOKEN \
  -f Dockerfile.sherpa-binary \
  -t openvidu/agent-speech-processing-sherpa-binary:main \
  .
```

**Option 2 - Using a token file:**
```bash
echo "ghp_your_token_here" > ~/.github_token
docker build \
  --secret id=github_token,src=$HOME/.github_token \
  -f Dockerfile.sherpa-binary \
  -t openvidu/agent-speech-processing-sherpa-binary:main \
  .
```

## Verification

After building, you can verify that the binary was created correctly:

```bash
# Run the container interactively
docker run -it --entrypoint /bin/bash openvidu/agent-speech-processing-sherpa-binary:main

# Inside the container:
ls -la /app/agent_dist/main  # The compiled binary
ls /app/*.py  # Should not exist - no Python source files!
```

## Security Considerations

### What's Protected

1. **Application Logic**: The main agent logic in `main.py`, `stt_impl.py`, and `vad_stt_wrapper.py` is compiled to native code
2. **Utility Code**: The `openviduagentutils` package is also bundled in the binary
3. **Business Logic**: STT provider selection, configuration, and processing logic is hidden

### What's Not Protected

1. **Configuration Files**: Any YAML or environment-based configuration is still readable
2. **Models**: ML models (Sherpa, VAD) are still present as files (they need to be loaded at runtime)
3. **Network Traffic**: The LiveKit protocol communication can still be observed
4. **Runtime Behavior**: The binary can still be analyzed through debugging/profiling tools

### Important Notes

- **No protection is absolute**: While Nuitka makes reverse engineering significantly harder, a determined attacker with sufficient time and resources can still analyze the binary
- **Security in depth**: This binarization should be one layer in a comprehensive security strategy
- **Complementary measures**: Consider also using:
  - Network encryption (already provided by LiveKit)
  - Access control and authentication
  - Runtime monitoring and intrusion detection
  - Regular security updates

## Performance Impact

Nuitka compilation typically provides:
- **Startup Time**: Slightly faster than interpreted Python (pre-compiled code)
- **Memory Usage**: Similar or slightly higher (bundled interpreter)
- **Runtime Performance**: Similar to CPython for most code, with potential improvements for some operations
- **Image Size**: Larger than standard Python image due to bundled dependencies, but acceptable for production use

## Troubleshooting

### Binary Fails to Start

If the binary fails to start, check:
1. All required system libraries are installed in the runtime image
2. File permissions are correct (`chmod +x` if needed)
3. Models directory exists and contains the required files

### Missing Module Errors

If you see "ModuleNotFoundError" at runtime:
1. Add the missing module with `--include-module=module_name` in the Nuitka compilation step
2. For packages, use `--include-package=package_name`
3. Rebuild the Docker image

### Dynamic Imports

If the agent uses dynamic imports (e.g., `importlib.import_module()`), you may need to:
1. Identify the dynamically imported modules
2. Add them explicitly with `--include-module` or `--include-package`
3. Alternatively, restructure code to use static imports where possible

## Alternative Approaches Considered

### Cython
- **Pros**: Fine-grained control, C-level optimizations possible
- **Cons**: Requires maintaining separate .pyx files or heavy annotation, main entry point still Python, less complete obfuscation
- **Verdict**: More complex to set up and maintain for this use case

### PyInstaller/PyOxidizer
- **Pros**: Popular, well-documented
- **Cons**: Easier to reverse engineer (just unpacks Python bytecode), not true compilation
- **Verdict**: Provides less security than Nuitka

### PyArmor
- **Pros**: Focused on obfuscation
- **Cons**: Proprietary/commercial for serious use, can impact performance, still Python bytecode
- **Verdict**: Not truly binarization, licensing concerns

## Future Improvements

Potential enhancements to consider:

1. **Further Optimization**: Use `--onefile` mode to create a single executable file instead of a distribution directory
2. **Strip Debug Symbols**: Add `--strip` and `--no-debug` flags for smaller binaries
3. **Static Linking**: Explore static linking of dependencies to reduce runtime dependencies
4. **Cross-Compilation**: Build for different architectures (ARM64, x86_64) in CI/CD

## References

- [Nuitka Official Documentation](https://nuitka.net/doc/user-manual.html)
- [Nuitka User Manual](https://nuitka.net/user-documentation/user-manual.html)
- [LiveKit Agents Framework](https://docs.livekit.io/agents/)
- [Docker BuildKit Secrets](https://docs.docker.com/build/building/secrets/)
