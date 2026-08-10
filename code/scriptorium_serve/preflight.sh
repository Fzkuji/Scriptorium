#!/bin/bash
# Everything that has to be true before the platform is pointed at this.
# Each line is a run that died: a service that was not up, a tunnel that had
# closed, a chunk refused over a blank turn in it, a writer with no credit.
set -u
URL=http://23.148.204.248:8600
CONFIG=/Users/fzkuji/.config/scriptorium
T=$(cat $CONFIG/leaderboard-service-token.txt)
fail=0
say() { printf "%-32s %s\n" "$1" "$2"; }
bad() { say "$1" "$2"; fail=1; }

code=$(curl -s -m 25 -o /dev/null -w "%{http_code}" $URL/health)
[ "$code" = 200 ] && say "registered URL" "ok ($code)" || bad "registered URL" "FAIL ($code)"

live=$(pgrep -f "ssh -N .*:860" | wc -l | tr -d ' ')
[ "$live" -ge 3 ] && say "ssh tunnels" "$live of 4 up" || bad "ssh tunnels" "FAIL ($live of 4)"

pgrep -f "uvicorn scriptorium_serve" >/dev/null \
  && say "memory service" "up" || bad "memory service" "FAIL"

# 18067 Adds at the measured $0.0042 each is what one full run costs.
left=$(curl -s -m 25 -H "Authorization: Bearer $(cat $CONFIG/openrouter-key.txt)" \
  https://openrouter.ai/api/v1/credits \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print(f\"{d['total_credits']-d['total_usage']:.2f}\")")
python3 -c "import sys; sys.exit(0 if float('$left') >= 76 else 1)" \
  && say "writer credit" "ok (\$$left of \$76 needed)" \
  || bad "writer credit" "FAIL (\$$left, a run needs \$76)"

cd /Users/fzkuji/scriptorium-runs/wt-competition || exit 1
out=$(PYTHONPATH=code python3 -m scriptorium_serve.smoke --base-url $URL --token "$T" 2>&1)
echo "$out" | grep -q "All contract checks passed" \
  && say "contract (smoke)" "ok" || { bad "contract (smoke)" "FAIL"; echo "$out" | tail -5; }

blank=$(curl -s -m 120 -o /dev/null -w "%{http_code}" -X POST $URL/add \
  -H "Content-Type: application/json" -H "Authorization: Bearer $T" \
  -d '{"request_id":"preflight-blank","user_id":"preflight-blank","session_id":"s",
       "messages":[{"role":"user","content":"Preflight sentence.","timestamp":1700000000000},
                   {"role":"assistant","content":"   ","timestamp":1700000001000}]}')
[ "$blank" = 200 ] && say "blank turn in a chunk" "ok (200)" \
  || bad "blank turn in a chunk" "FAIL ($blank)"
rm -rf /Users/fzkuji/scriptorium-runs/live-workspaces/preflight-blank

say "memory for the resumed run" \
  "$(ls /Users/fzkuji/scriptorium-runs/live-workspaces | grep -c c6d028e8) workspaces"

echo
if [ $fail = 0 ]; then
  echo "READY. Ten minutes of load is still the last word before launching:"
else
  echo "NOT READY. The FAILs above come first."
fi
cat <<'EOF'
  cd /Users/fzkuji/scriptorium-runs/wt-competition
  PYTHONPATH=code python3 -m scriptorium_serve.loadtest \
    --base-url http://23.148.204.248:8600 \
    --token "$(cat ~/.config/scriptorium/leaderboard-service-token.txt)" \
    --workers 16 --minutes 10 --min-chars 3000 --clean
EOF
exit $fail
