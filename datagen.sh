#!/bin/bash

# Create the test-data folder in the same directory as data
mkdir -p test-data/truth
mkdir -p test-data/transformed

# Counter for new numbering
counter=0

# Move the last 3000 images from the truth folder
for i in {20488..25609}; do
    mv "data/truth/$i.png" "test-data/truth/$counter.png"
    counter=$((counter + 1))
done

# Reset counter
counter=0

# Move the last 3000 images from the transformed folder
for i in {20488..25609}; do
    mv "data/transformed/$i.png" "test-data/transformed/$counter.png"
    counter=$((counter + 1))
done

echo "Images moved to test-data folder and renumbered."
