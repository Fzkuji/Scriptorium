#!/bin/bash
# Reverse tunnels from this machine to the VPS that owns the registered URL.
#
# They replace a Cloudflare quick tunnel, which capped every request at a
# hundred seconds: Cloudflare answers 524 when an origin takes longer, and a
# cut-off Add is reported back as ADD_API_CONTRACT_MISMATCH, which ends the
# run. Plain SSH forwarding has no such ceiling.
#
# Four of them, not one. A single connection carried every request, and under
# load the far side closed it twice inside two minutes; each close took the
# whole service off the air until the loop below redialled. Four connections
# make one closing a quarter of the capacity, and nginx keeps the other three
# serving while it comes back.
REMOTE=root@23.148.204.248
PORTS="8601 8602 8603 8604"

hold() {
  local port=$1
  while true; do
    ssh -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o TCPKeepAlive=yes \
      -o StrictHostKeyChecking=no \
      -R "127.0.0.1:${port}:127.0.0.1:8600" \
      "$REMOTE"
    echo "$(date -u +%FT%TZ) tunnel :$port dropped, reconnecting" >&2
    sleep 2
  done
}

for port in $PORTS; do
  hold "$port" &
done
wait
