
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class MockGeminiHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"Mock received POST to {self.path}")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        # Standard Gemini streaming response (JSON array)
        response = [
            {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "from the mock!"}]}}]},
        ]
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(port=8341):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MockGeminiHandler)
    print(f"Starting mock Gemini upstream on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
