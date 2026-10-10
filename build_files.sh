#!/usr/bin/env bash
# Build script for Vercel deployment
echo "Building VidNote AI for Vercel..."
python3 -m pip install -r requirements.txt
python3 manage.py collectstatic --noinput --clear
echo "Build completed successfully!"
