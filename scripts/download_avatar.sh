#!/bin/bash
# Download a free rigged humanoid avatar for the pipeline
# This is the official Khronos glTF sample "Fox" model which has a full armature

echo "Downloading free rigged avatar..."

# Option 1: Ready Player Me sample avatar (rigged humanoid with visemes)
wget -q --show-progress \
    "https://models.readyplayer.me/64bfa15f0e72c63d7c3934a6.glb" \
    -O avatars/default_avatar.glb 2>/dev/null

if [ -f avatars/default_avatar.glb ] && [ -s avatars/default_avatar.glb ]; then
    echo "Downloaded Ready Player Me avatar to avatars/default_avatar.glb"
    exit 0
fi

# Option 2: Mixamo-rigged model from GitHub
wget -q --show-progress \
    "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb" \
    -O avatars/default_avatar.glb 2>/dev/null

if [ -f avatars/default_avatar.glb ] && [ -s avatars/default_avatar.glb ]; then
    echo "Downloaded X-Bot avatar to avatars/default_avatar.glb"
    exit 0
fi

echo "Auto-download failed. Please manually place a rigged .glb or .fbx in the avatars/ directory."
echo "Recommended sources:"
echo "  - https://readyplayer.me (create a free avatar)"
echo "  - https://www.mixamo.com (Adobe account required)"
echo "  - https://sketchfab.com/search?q=rigged+humanoid&type=models&sort_by=-likeCount"
exit 1
