#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

echo "Staging all changes..."
git add .

echo "Committing changes with message: \"$1\""
git commit -m "$1"

if [ $? -eq 0 ]; then
  echo "Pushing changes to remote..."
  git push
  if [ $? -eq 0 ]; then
    echo "Successfully committed and pushed."
  else
    echo "Error: Failed to push changes."
  fi
else
  echo "Error: Failed to commit changes. There might be nothing to commit or an issue with Git."
fi
