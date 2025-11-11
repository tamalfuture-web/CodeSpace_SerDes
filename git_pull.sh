#!/bin/bash

echo "Pulling latest changes from remote..."
git pull

if [ $? -eq 0 ]; then
  echo "Successfully pulled latest changes."
else
  echo "Error: Failed to pull changes."
fi
