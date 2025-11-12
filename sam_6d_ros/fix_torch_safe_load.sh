#!/bin/bash

# Find all tasks.py files and update torch.load calls to include weights_only=False
find . -name 'tasks.py' | while read -r file; do
    echo "Processing $file"
    # Use sed to add weights_only=False if not present
    sed -i -E "s/torch\.load\(([^)]*)\)/torch.load(\1, weights_only=False)/g" "$file"
done