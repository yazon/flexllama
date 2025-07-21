#!/bin/bash
# Debug script to find libmtmd.so in the build stage

echo "=== Searching for libmtmd.so in build directory ==="
find /build -name "libmtmd.so*" -type f 2>/dev/null | while read -r file; do
    echo "Found: $file"
    ls -la "$file"
    ldd "$file" 2>/dev/null || echo "  (ldd not available or not a valid library)"
done

echo -e "\n=== Searching for libmtmd.so in /usr/local ==="
find /usr/local -name "libmtmd.so*" -type f 2>/dev/null | while read -r file; do
    echo "Found: $file"
    ls -la "$file"
done

echo -e "\n=== Checking CMake install manifest ==="
if [ -f /build/llama.cpp/build/install_manifest.txt ]; then
    grep -i "mtmd" /build/llama.cpp/build/install_manifest.txt || echo "No mtmd entries in install manifest"
else
    echo "No install manifest found"
fi

echo -e "\n=== Checking if mtmd was built ==="
if [ -d /build/llama.cpp/build/tools/mtmd ]; then
    echo "mtmd build directory exists:"
    ls -la /build/llama.cpp/build/tools/mtmd/
else
    echo "mtmd build directory not found"
fi

echo -e "\n=== Checking llama-server dependencies ==="
if [ -f /usr/local/bin/llama-server ]; then
    echo "llama-server dependencies:"
    ldd /usr/local/bin/llama-server | grep -i mtmd || echo "  No direct mtmd dependency found"
fi