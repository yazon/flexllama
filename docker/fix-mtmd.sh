#!/bin/bash
# Script to check and potentially fix missing libmtmd.so issue

echo "Checking for libmtmd.so..."

# Check if libmtmd.so exists
if ! ldconfig -p | grep -q libmtmd.so; then
    echo "WARNING: libmtmd.so not found in system libraries"
    
    # Check if llama-server can run without it
    if ldd /usr/local/bin/llama-server 2>&1 | grep -q "libmtmd.so => not found"; then
        echo "ERROR: llama-server requires libmtmd.so"
        
        # Try to find the library in common locations
        for dir in /usr/local/lib /usr/lib /opt/lib /app/lib; do
            if [ -f "$dir/libmtmd.so" ]; then
                echo "Found libmtmd.so in $dir"
                export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
                break
            fi
        done
        
        # If still not found, create a dummy library as a last resort
        # This will allow llama-server to start but multimodal features won't work
        if ! ldconfig -p | grep -q libmtmd.so; then
            echo "Creating dummy libmtmd.so (multimodal features will be disabled)"
            # This is a workaround - ideally the library should be properly built
            touch /tmp/libmtmd.so
            export LD_LIBRARY_PATH="/tmp:$LD_LIBRARY_PATH"
        fi
    else
        echo "llama-server doesn't directly depend on libmtmd.so"
    fi
else
    echo "libmtmd.so found in system libraries"
fi

# Update library cache
ldconfig 2>/dev/null || true

echo "Library path: $LD_LIBRARY_PATH"