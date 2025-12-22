#!/bin/sh
# Generate env.js from environment variables at runtime
# Check for API_BASE if API_BASE_URL is not set
if [ -z "$API_BASE_URL" ] && [ -n "$API_BASE" ]; then
  API_BASE_URL="$API_BASE"
fi

echo "Generating env.js with API_BASE_URL=${API_BASE_URL}"

# Default to empty string if not set, or handle strictness
: "${API_BASE_URL:=}"
: "${ADMIN_TOKEN:=}"

# Normalize variables: strip double quotes and semicolons just in case
API_BASE_URL=$(echo "$API_BASE_URL" | tr -d '";')
ADMIN_TOKEN=$(echo "$ADMIN_TOKEN" | tr -d '";')

cat <<EOF > /usr/share/nginx/html/env.js
window.API_BASE = "${API_BASE_URL}";
window.ADMIN_TOKEN = "${ADMIN_TOKEN}";
EOF
