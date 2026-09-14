"""Authenticated loopback bridge for the route planner Site."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse
import webbrowser

from .engine import LiveEngine

SITE_ORIGIN = 'https://blackflow-guide.wzm5252429.chatgpt.site'
DEFAULT_PORT = 19761


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, engine, *, origins=(SITE_ORIGIN,), window_provider=None):
        if address[0] not in {'127.0.0.1', 'localhost'}:
            raise ValueError('Bridge must bind to loopback')
        self.engine = engine
        self.origins = frozenset(origins)
        self.token = secrets.token_urlsafe(32)
        self.pair_code = f'{secrets.randbelow(1_000_000):06d}'
        self.pair_expires = time.monotonic() + 600
        self.pair_attempts = 0
        self.auth_lock = threading.Lock()
        self.window_provider = window_provider or (lambda: [])
        super().__init__(address, BridgeHandler)


class BridgeHandler(BaseHTTPRequestHandler):
    server: BridgeServer
    protocol_version = 'HTTP/1.1'

    def setup(self):
        # Bound incomplete-body requests on this threaded loopback server.
        self.request.settimeout(10)
        super().setup()

    def log_message(self, *_):
        pass  # Never write pairing codes, tokens or request bodies to access logs.

    def _headers(self, status, content_type, length):
        self.send_response(status)
        origin = self.headers.get('Origin', '')
        if origin in self.server.origins:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))

    def _reply(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self._headers(status, 'application/json; charset=utf-8', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _origin_valid(self):
        if len(self.headers.get_all('Host', [])) != 1 or len(self.headers.get_all('Origin', [])) > 1:
            return False
        host = self.headers.get('Host', '')
        valid_host = host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        origin = self.headers.get('Origin')
        # Non-browser clients may omit Origin. Sensitive endpoints still require
        # a bearer; health is public and pairing requires a one-time local code.
        return valid_host and (origin is None or origin in self.server.origins)

    def _authorized(self):
        if len(self.headers.get_all('Authorization', [])) != 1:
            return False
        auth = self.headers.get('Authorization', '')
        return self._origin_valid() and hmac.compare_digest(auth.encode('utf-8'), ('Bearer ' + self.server.token).encode('ascii'))

    def do_OPTIONS(self):
        if not self._origin_valid() or self.headers.get('Origin') not in self.server.origins:
            return self._reply(403, {'error': '不允许此网站连接本机接管器'})
        self._headers(204, 'text/plain', 0)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
        self.send_header('Access-Control-Max-Age', '300')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._origin_valid():
            return self._reply(403, {'error': '来源或主机不受信任'})
        if path == '/v1/health':
            return self._reply(200, {'service': 'blackflow-live', 'version': 1, 'pairing_required': True})
        if not self._authorized():
            return self._reply(401, {'error': '请使用本机启动器的连接码进行配对'})
        if path == '/v1/status':
            return self._reply(200, self.server.engine.status(heartbeat=True))
        if path == '/v1/windows':
            try:
                return self._reply(200, {'windows': self.server.window_provider()})
            except Exception as exc:
                return self._reply(503, {'error': str(exc)})
        if path == '/v1/frame':
            frame = self.server.engine.preview()
            if frame is None:
                return self._reply(404, {'error': '尚未取得游戏截图'})
            self._headers(200, 'image/jpeg', len(frame))
            self.end_headers()
            return self.wfile.write(frame)
        return self._reply(404, {'error': '未找到接口'})

    def do_POST(self):
        if not self._origin_valid():
            self.close_connection = True
            return self._reply(403, {'error': '来源或主机不受信任'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self.close_connection = True
            return self._reply(415, {'error': '需要 application/json'})
        try:
            if len(self.headers.get_all('Content-Length', [])) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError('不支持重复长度或传输编码')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('请求长度无效')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('请求必须是对象')
        except (ValueError, UnicodeError):
            self.close_connection = True
            return self._reply(400, {'error': '请求格式无效'})
        path = urlparse(self.path).path
        if path == '/v1/pair':
            with self.server.auth_lock:
                self.server.pair_attempts += 1
                code = data.get('code')
                code_valid = isinstance(code, str) and len(code) == 6 and code.isascii() and code.isdigit()
                valid = (self.server.pair_attempts <= 8 and time.monotonic() < self.server.pair_expires
                         and code_valid and hmac.compare_digest(code, self.server.pair_code))
                if not valid:
                    return self._reply(403, {'error': '连接码不正确或已过期，请重新启动本机接管器'})
                self.server.pair_expires = 0  # Single use; tokens live only for this process.
                return self._reply(200, {'token': self.server.token})
        if not self._authorized():
            return self._reply(401, {'error': '连接已失效，请重新配对本机接管器'})
        try:
            if path in {'/v1/start', '/v1/observe'}:
                if data.get('ending', 'first') != 'first':
                    return self._reply(400, {'error': '当前接管仅支持一结局'})
                hwnd = data.get('hwnd')
                if hwnd is not None and (type(hwnd) is not int or hwnd <= 0):
                    return self._reply(400, {'error': '窗口句柄无效'})
                result = self.server.engine.start(hwnd, observe_only=path == '/v1/observe')
            elif path == '/v1/pause':
                result = self.server.engine.pause()
            elif path == '/v1/resume':
                result = self.server.engine.resume()
            elif path == '/v1/stop':
                result = self.server.engine.stop()
            else:
                return self._reply(404, {'error': '未找到接口'})
            return self._reply(200, result)
        except ValueError as exc:
            return self._reply(409, {'error': str(exc)})
        except Exception as exc:
            return self._reply(500, {'error': str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description='黑流树海本机接管器')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--runtime', type=Path, help='MAA / BFMapRecognizer 运行目录')
    parser.add_argument('--open-site', action='store_true')
    parser.add_argument('--allow-origin', action='append', default=[], help='显式允许的开发网站来源')
    args = parser.parse_args(argv)
    from .runtime import GameRuntime, discover_windows
    engine = LiveEngine(lambda hwnd: GameRuntime(hwnd=hwnd, runtime=args.runtime))
    server = BridgeServer(('127.0.0.1', args.port), engine,
                          origins=(SITE_ORIGIN, *args.allow_origin), window_provider=discover_windows)
    print(f'本机接管器已启动：http://127.0.0.1:{args.port}', flush=True)
    print(f'网站连接码（10 分钟内有效）：{server.pair_code}', flush=True)
    print('点击网站「自动路线决策」开始；Esc 或鼠标移到桌面左上角暂停。', flush=True)
    if args.open_site:
        webbrowser.open(SITE_ORIGIN + '/#planner?bridge_token=' + server.token)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()


if __name__ == '__main__':
    main()
