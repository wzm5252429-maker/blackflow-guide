"""Loopback HTTP integration tests with a fake engine and no game access."""
import http.client
import json
import threading
import time
import unittest

from blackflow_live.service import BridgeServer, SITE_ORIGIN


class FakeEngine:
    def __init__(self):
        self.active = False
        self.calls = []
        self.state = 'idle'
        self.frame = b'fake-jpeg'

    def status(self, *, heartbeat=False):
        self.calls.append(('status', heartbeat))
        return {'state': self.state, 'clicks': 0}

    def preview(self):
        self.calls.append(('preview',))
        return self.frame

    def start(self, hwnd=None, *, observe_only=False):
        if self.active:
            raise ValueError('已有接管会话')
        self.active = True
        self.state = 'observing' if observe_only else 'running'
        self.calls.append(('start', hwnd, observe_only))
        return {'state': self.state}

    def pause(self):
        self.state = 'paused'
        self.calls.append(('pause',))
        return {'state': self.state}

    def resume(self):
        if not self.active:
            raise ValueError('会话已结束')
        self.state = 'running'
        self.calls.append(('resume',))
        return {'state': self.state}

    def stop(self):
        self.active = False
        self.state = 'stopped'
        self.calls.append(('stop',))
        return {'state': self.state}


class LiveServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = BridgeServer(('127.0.0.1', 0), self.engine,
                                   window_provider=lambda: [{'hwnd': 123, 'title': '明日方舟'}])
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, *, auth=False, origin=SITE_ORIGIN, headers=None, body=None):
        request_headers = {}
        if origin is not None:
            request_headers['Origin'] = origin
        if auth:
            request_headers['Authorization'] = 'Bearer ' + self.server.token
        if data is not None:
            body = json.dumps(data).encode('utf-8')
            request_headers['Content-Type'] = 'application/json'
        if headers:
            request_headers.update(headers)
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            result = response.read()
            return response.status, dict(response.getheaders()), result
        finally:
            connection.close()

    def test_only_loopback_bind_allowed(self):
        with self.assertRaisesRegex(ValueError, 'loopback'):
            BridgeServer(('0.0.0.0', 0), FakeEngine())

    def test_health_public_but_contains_no_credentials(self):
        status, headers, body = self.request('GET', '/v1/health', origin=None)
        self.assertEqual(status, 200)
        self.assertNotIn(self.server.token.encode(), body)
        self.assertNotIn(self.server.pair_code.encode(), body)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertFalse(self.engine.calls)

    def test_origin_and_host_are_strict_even_with_correct_bearer(self):
        for origin in ('https://evil.example', 'null', SITE_ORIGIN + '.evil.example', SITE_ORIGIN + '/'):
            status, headers, _ = self.request('GET', '/v1/status', auth=True, origin=origin)
            self.assertEqual(status, 403)
            self.assertNotIn('Access-Control-Allow-Origin', headers)
        for host in ('evil.example', '127.0.0.1', f'127.0.0.1:{self.server.server_port}.evil.example'):
            status, _, _ = self.request('GET', '/v1/status', auth=True, headers={'Host': host})
            self.assertEqual(status, 403)
        self.assertFalse(self.engine.calls)

    def test_duplicate_host_headers_rejected(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.putrequest('GET', '/v1/status', skip_host=True)
            connection.putheader('Host', f'127.0.0.1:{self.server.server_port}')
            connection.putheader('Host', 'evil.example')
            connection.putheader('Authorization', 'Bearer ' + self.server.token)
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
        finally:
            connection.close()
        self.assertFalse(self.engine.calls)

    def test_pairing_code_is_single_use_and_token_allows_status(self):
        status, _, body = self.request('POST', '/v1/pair', {'code': self.server.pair_code})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['token'], self.server.token)
        status, _, _ = self.request('POST', '/v1/pair', {'code': self.server.pair_code})
        self.assertEqual(status, 403)
        status, _, _ = self.request('GET', '/v1/status', auth=True)
        self.assertEqual(status, 200)
        self.assertIn(('status', True), self.engine.calls)

    def test_invalid_nonascii_pairing_code_returns_error_without_crashing(self):
        for code in ('', '错误连接码', 123456, True, None, '１２３４５６'):
            with self.subTest(code=code):
                status, _, body = self.request('POST', '/v1/pair', {'code': code})
                self.assertEqual(status, 403)
                self.assertIn('error', json.loads(body))

    def test_pairing_code_expiration_and_attempt_limit(self):
        self.server.pair_attempts = 8
        status, _, _ = self.request('POST', '/v1/pair', {'code': self.server.pair_code})
        self.assertEqual(status, 403)
        self.server.pair_attempts = 0
        self.server.pair_expires = time.monotonic() - 1
        status, _, _ = self.request('POST', '/v1/pair', {'code': self.server.pair_code})
        self.assertEqual(status, 403)

    def test_screenshot_windows_status_and_mutations_require_auth(self):
        for path in ('/v1/frame', '/v1/windows', '/v1/status'):
            status, _, body = self.request('GET', path)
            self.assertEqual(status, 401)
            self.assertNotIn(b'fake-jpeg', body)
        for path in ('/v1/start', '/v1/observe', '/v1/pause', '/v1/resume', '/v1/stop'):
            status, _, _ = self.request('POST', path, {})
            self.assertEqual(status, 401)
        self.assertFalse(self.engine.calls)

    def test_bad_or_nonascii_bearer_fails_without_crashing(self):
        for token in ('Bearer wrong', 'bearer ' + self.server.token, 'Bearer \xe9'):
            status, _, _ = self.request('GET', '/v1/status', headers={'Authorization': token})
            self.assertEqual(status, 401)
        self.assertFalse(self.engine.calls)

    def test_bearer_without_origin_works_for_local_clients(self):
        status, _, _ = self.request('GET', '/v1/status', auth=True, origin=None)
        self.assertEqual(status, 200)

    def test_private_network_preflight_only_for_trusted_origin(self):
        status, headers, _ = self.request('OPTIONS', '/v1/start', headers={
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization,content-type',
            'Access-Control-Request-Private-Network': 'true',
        })
        self.assertEqual(status, 204)
        self.assertEqual(headers['Access-Control-Allow-Origin'], SITE_ORIGIN)
        self.assertEqual(headers['Access-Control-Allow-Private-Network'], 'true')
        self.assertIn('Authorization', headers['Access-Control-Allow-Headers'])
        for origin in ('https://evil.example', None):
            status, headers, _ = self.request('OPTIONS', '/v1/start', origin=origin)
            self.assertEqual(status, 403)
            self.assertNotIn('Access-Control-Allow-Private-Network', headers)

    def test_start_pause_resume_stop_and_single_session(self):
        for path, payload, expected in (
            ('/v1/start', {'hwnd': 123, 'ending': 'first'}, 200),
            ('/v1/start', {}, 409),
            ('/v1/observe', {}, 409),
            ('/v1/pause', {}, 200),
            ('/v1/resume', {}, 200),
            ('/v1/stop', {}, 200),
            ('/v1/resume', {}, 409),
            ('/v1/observe', {'hwnd': 123}, 200),
        ):
            with self.subTest(path=path, expected=expected):
                status, _, _ = self.request('POST', path, payload, auth=True)
                self.assertEqual(status, expected)
        self.assertEqual(self.engine.calls, [('start', 123, False), ('pause',), ('resume',), ('stop',), ('start', 123, True)])

    def test_only_first_ending_and_valid_integer_hwnd_accepted(self):
        for data in ({'ending': 'second'}, {'ending': None}, {'hwnd': True}, {'hwnd': -1}, {'hwnd': '123'}, {'hwnd': 1.5}):
            status, _, _ = self.request('POST', '/v1/start', data, auth=True)
            self.assertEqual(status, 400)
        self.assertFalse(self.engine.calls)

    def test_invalid_json_types_content_type_and_lengths(self):
        for body in (b'{', b'[]', b'null', b'"text"', b'\xff', b'{"x":' + b'0' * 4100 + b'}'):
            status, _, _ = self.request('POST', '/v1/start', auth=True, body=body,
                                         headers={'Content-Type': 'application/json'})
            self.assertEqual(status, 400)
        status, _, _ = self.request('POST', '/v1/start', auth=True, body=b'{}', headers={'Content-Type': 'text/plain'})
        self.assertEqual(status, 415)
        for headers in ({'Content-Length': '-1'}, {'Content-Length': 'invalid'}, {'Transfer-Encoding': 'chunked'}):
            status, _, _ = self.request('POST', '/v1/start', {}, auth=True, headers=headers)
            self.assertEqual(status, 400)
        self.assertFalse(self.engine.calls)

    def test_frame_no_store_and_unavailable_frame(self):
        status, headers, body = self.request('GET', '/v1/frame', auth=True)
        self.assertEqual((status, body), (200, b'fake-jpeg'))
        self.assertEqual(headers['Content-Type'], 'image/jpeg')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.engine.frame = None
        status, _, _ = self.request('GET', '/v1/frame', auth=True)
        self.assertEqual(status, 404)

    def test_window_listing_and_failure_response(self):
        status, _, body = self.request('GET', '/v1/windows', auth=True)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['windows'][0]['hwnd'], 123)
        def unavailable():
            raise RuntimeError('Unavailable game')
        self.server.window_provider = unavailable
        status, _, _ = self.request('GET', '/v1/windows', auth=True)
        self.assertEqual(status, 503)

    def test_unknown_endpoint_does_not_call_engine(self):
        for method in ('GET', 'POST'):
            status, _, _ = self.request(method, '/v1/unknown', {} if method == 'POST' else None, auth=True)
            self.assertEqual(status, 404)
        self.assertFalse(self.engine.calls)


if __name__ == '__main__':
    unittest.main()
