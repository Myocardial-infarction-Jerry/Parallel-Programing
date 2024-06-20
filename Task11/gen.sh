#!/bin/bash

# Create the build directory
mkdir -p build
if [ $? -ne 0 ]; then
    echo "Failed to create build directory"
    exit 1
fi

# Change into the build directory
cd build
if [ $? -ne 0 ]; then
    echo "Failed to change into build directory"
    exit 1
fi

# Run cmake and make
cmake .. && make
if [ $? -ne 0 ]; then
    echo "Failed to run cmake or make"
    exit 1
fi

# Change back to the original directory
cd ..
if [ $? -ne 0 ]; then
    echo "Failed to change back to original directory"
    exit 1
fi

# Check the current working directory
if [ "$(pwd)" != "$(dirname "$(realpath "$0")")" ]; then
    echo "Current working directory is not the original directory"
    exit 1
fi

echo "Script completed successfully"