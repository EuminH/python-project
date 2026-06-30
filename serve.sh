#!/usr/bin/env bash
# Launch the dashboard locally AND expose it on a public URL via Cloudflare Tunnel.
#
# Requires cloudflared:  brew install cloudflared
# The public *.trycloudflare.com URL is TEMPORARY — it changes every run and stays
# live only while this script (and your Mac, awake) keeps running. Ctrl+C to stop.
#
# For a permanent 24/7 URL, deploy to Streamlit Community Cloud instead (see README).
set -e

PORT=8501
STREAMLIT=/opt/anaconda3/bin/streamlit   # adjust if your Python lives elsewhere

echo "▶ Starting dashboard on http://localhost:$PORT ..."
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
"$STREAMLIT" run app.py \
  --server.headless true --server.port "$PORT" \
  --server.enableCORS false --server.enableXsrfProtection false \
  > /tmp/streamlit_log.txt 2>&1 &

sleep 8
echo "▶ Opening public tunnel (share the https://….trycloudflare.com link it prints) ..."
echo "  Press Ctrl+C to take the site offline."
cloudflared tunnel --url "http://localhost:$PORT"
