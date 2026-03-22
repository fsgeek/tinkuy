#!/bin/bash

# 1. Start mock Gemini upstream
python3 scripts/mock_gemini_upstream.py > mock.log 2>&1 &
MOCK_PID=$!
echo "Mock upstream started (PID $MOCK_PID) on 8341"

# 2. Start Tinkuy gateway pointed at mock
export GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8341
python3 -m tinkuy serve --port 8342 > gateway.log 2>&1 &
GW_PID=$!
echo "Tinkuy gateway started (PID $GW_PID) on 8342"

# Wait for startup
sleep 3

# 3. Send test request
echo "Sending test request to Tinkuy..."
curl -s -X POST "http://127.0.0.1:8342/v1beta/models/gemini-1.5-pro:streamGenerateContent" \
     -H "Content-Type: application/json" \
     -H "x-tinkuy-session: loopback-test" \
     -d '{
       "contents": [{"role": "user", "parts": [{"text": "hi there"}]}]
     }' > response.json

echo "Response received:"
cat response.json
echo ""

# 4. Check Tinkuy status
echo "Checking Tinkuy status..."
curl -s -H "x-tinkuy-session: loopback-test" "http://127.0.0.1:8342/v1/tinkuy/status" > status.json
cat status.json
echo ""

# 5. Cleanup
kill $GW_PID $MOCK_PID
echo "Cleanup complete."

# 6. Verification
if grep -q "from the mock!" response.json && grep -q '"turn":1' status.json; then
    echo "SUCCESS: Gemini integration verified end-to-end!"
    exit 0
else
    echo "FAILURE: Integration test failed. Check mock.log and gateway.log"
    # Print logs if failed
    echo "--- GATEWAY LOG ---"
    cat gateway.log
    echo "--- MOCK LOG ---"
    cat mock.log
    exit 1
fi
