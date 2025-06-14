#!/bin/bash
mkdir -p models/whisper-large-v3

echo "Downloading Whisper Large-v3..."
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors -o models/whisper-large-v3/model.safetensors
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/config.json -o models/whisper-large-v3/config.json
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json -o models/whisper-large-v3/tokenizer.json
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/preprocessor_config.json -o models/whisper-large-v3/preprocessor_config.json
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/vocabulary.json -o models/whisper-large-v3/vocabulary.json
curl -L https://huggingface.co/openai/whisper-large-v3/resolve/main/merges.txt -o models/whisper-large-v3/merges.txt

echo "✓ Download complete!"