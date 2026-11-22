# -*- coding: utf-8 -*-
"""
本地 Demo 网页服务：http://localhost:8000
用法：python app.py
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rag

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS, EMBS = rag.build_index(rag.load_chunks())
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(BASE, "index.html"), encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}, ensure_ascii=False))

    def do_POST(self):
        if self.path != "/api/ask":
            self._send(404, json.dumps({"error": "not found"}, ensure_ascii=False))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = (payload.get("question") or "").strip()
            if not question:
                self._send(400, json.dumps({"error": "empty question"}, ensure_ascii=False))
                return
            answer, hits = rag.answer(question, CHUNKS, EMBS)
            self._send(200, json.dumps({"answer": answer, "sources": hits}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    print("已索引 %d 个文本块，服务启动： http://%s:%d" % (len(CHUNKS), HOST, PORT))
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
