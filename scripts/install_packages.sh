#!/usr/bin/env bash
set -euo pipefail

echo "Make sure conda env is already active"
echo "Installing ultralytics with --no-deps ..."
pip install --no-deps "ultralytics==8.3.234"

echo "ultralytics installed. Running quick check..."
python check_env.py || {
  echo "Environment check failed."
  exit 1
}