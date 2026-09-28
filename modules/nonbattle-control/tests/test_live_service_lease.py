"""Per-page lease integration tests with fake engines and loopback HTTP only."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import unittest

from blackflow_live.service import CLIENT_HEADER, LEASE_PROTOCOL
from tests import test_live_service as support

OWNER = 'page-owner-000000000001'
OBSERVER = 'page-observer-000000001'


class PageLeaseTests(unittest.TestCase):
    setUp = support.LiveServiceTests.setUp
    tearDown = support.LiveServiceTests.tearDown
    request = support.LiveServiceTests.request

    def start(self, client_id=OWNER, path='/v1/start'):
        return self.request('POST', path, {'client_id': client_id}, auth=True)

    def status(self, client_id=None, **kwargs):
        headers = {CLIENT_HEADER: client_id} if client_id is not None else None
        return self.request('GET', '/v1/status', auth=True, headers=headers, **kwargs)

    def test_idle_polling_never_acquires_or_renews_a_lease(self):
        for client_id in (None, OWNER, OBSERVER):
            status, _, body = self.status(client_id)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['lease'], {
                'protocol': LEASE_PROTOCOL, 'has_owner': False, 'is_owner': False,
            })
        self.assertIsNone(self.server.owner_client_id)
        self.assertNotIn(('status', True), self.engine.calls)

    def test_only_starting_page_status_renews_running_lease(self):
        status, _, body = self.start()
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)['lease']['is_owner'])
        self.engine.calls.clear()
        for client_id in (None, OBSERVER):
            status, _, body = self.status(client_id)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['lease'], {
                'protocol': LEASE_PROTOCOL, 'has_owner': True, 'is_owner': False,
            })
        self.assertNotIn(('status', True), self.engine.calls)
        status, _, body = self.status(OWNER)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)['lease']['is_owner'])
        self.assertEqual(self.engine.calls.count(('status', True)), 1)
        self.assertNotIn(OWNER.encode(), body)  # Responses need no transferable page ID.

    def test_observation_session_lease_also_belongs_to_its_starting_page(self):
        status, _, body = self.start(path='/v1/observe')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['state'], 'observing')
        self.engine.calls.clear()
        self.status(OBSERVER)
        self.assertNotIn(('status', True), self.engine.calls)
        self.status(OWNER)
        self.assertEqual(self.engine.calls.count(('status', True)), 1)

    def test_legacy_start_observe_resume_requires_protocol_upgrade(self):
        for path in ('/v1/start', '/v1/observe', '/v1/resume'):
            status, _, body = self.request('POST', path, {}, auth=True)
            self.assertEqual(status, 428)
            self.assertEqual(json.loads(body)['lease_protocol'], LEASE_PROTOCOL)
        self.assertFalse(self.engine.calls)

    def test_invalid_page_ids_do_not_reach_engine(self):
        invalid = (None, True, 123, '', 'short', 'a' * 129, '页面标识符' * 5, 'x' * 16 + '?', ['a' * 16])
        for path in ('/v1/start', '/v1/observe', '/v1/resume'):
            for value in invalid:
                with self.subTest(path=path, value=value):
                    status, _, _ = self.request('POST', path, {'client_id': value}, auth=True)
                    self.assertEqual(status, 400)
        for value in ('', 'short', 'x' * 129, 'x' * 16 + '?'):
            status, _, _ = self.status(value)
            self.assertEqual(status, 400)
        self.assertFalse(self.engine.calls)

    def test_duplicate_client_headers_are_rejected_without_heartbeat(self):
        self.start()
        self.engine.calls.clear()
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.putrequest('GET', '/v1/status')
            connection.putheader('Authorization', 'Bearer ' + self.server.token)
            connection.putheader(CLIENT_HEADER, OWNER)
            connection.putheader(CLIENT_HEADER, OBSERVER)
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()
        self.assertFalse(self.engine.calls)

    def test_page_id_in_query_does_not_renew_lease(self):
        self.start()
        self.engine.calls.clear()
        status, _, _ = self.request('GET', '/v1/status?client_id=' + OWNER, auth=True)
        self.assertEqual(status, 200)
        self.assertNotIn(('status', True), self.engine.calls)

    def test_page_id_never_replaces_bearer_origin_or_host_authorization(self):
        self.start()
        self.engine.calls.clear()
        status, _, _ = self.request('GET', '/v1/status', headers={CLIENT_HEADER: OWNER})
        self.assertEqual(status, 401)
        status, _, _ = self.status(OWNER, origin='https://evil.example')
        self.assertEqual(status, 403)
        status, _, _ = self.request('GET', '/v1/status', auth=True,
                                    headers={CLIENT_HEADER: OWNER, 'Host': 'evil.example'})
        self.assertEqual(status, 403)
        self.assertFalse(self.engine.calls)

    def test_running_resume_and_failed_start_cannot_steal_or_renew_lease(self):
        self.start()
        self.engine.calls.clear()
        for client_id in (OWNER, OBSERVER):
            status, _, _ = self.request('POST', '/v1/resume', {'client_id': client_id}, auth=True)
            self.assertEqual(status, 409)
        status, _, _ = self.start(OBSERVER)
        self.assertEqual(status, 409)
        self.assertEqual(self.server.owner_client_id, OWNER)
        self.assertNotIn(('resume',), self.engine.calls)
        self.assertNotIn(('status', True), self.engine.calls)

    def test_any_paired_page_may_pause_then_explicit_resume_transfers_ownership(self):
        self.start()
        status, _, body = self.request('POST', '/v1/pause', {}, auth=True)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['lease']['has_owner'])
        self.engine.calls.clear()
        self.status(OWNER)
        self.status(OBSERVER)
        self.assertNotIn(('status', True), self.engine.calls)
        status, _, body = self.request('POST', '/v1/resume', {'client_id': OBSERVER}, auth=True)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)['lease']['is_owner'])
        self.assertEqual(self.server.owner_client_id, OBSERVER)
        self.engine.calls.clear()
        self.status(OWNER)
        self.assertNotIn(('status', True), self.engine.calls)
        self.status(OBSERVER)
        self.assertEqual(self.engine.calls.count(('status', True)), 1)

    def test_automatic_pause_releases_owner_and_status_cannot_resume(self):
        self.start()
        self.engine.state = 'paused'
        self.engine.calls.clear()
        status, _, body = self.status(OWNER)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['state'], 'paused')
        self.assertFalse(json.loads(body)['lease']['has_owner'])
        self.assertIsNone(self.server.owner_client_id)
        self.assertNotIn(('status', True), self.engine.calls)
        self.assertNotIn(('resume',), self.engine.calls)

    def test_stop_and_terminal_states_release_owner(self):
        for terminal in ('idle', 'stopped', 'error', 'completed'):
            with self.subTest(terminal=terminal):
                self.engine.active = False
                self.start()
                self.engine.state = terminal
                self.engine.calls.clear()
                status, _, body = self.status(OWNER)
                self.assertEqual(status, 200)
                self.assertFalse(json.loads(body)['lease']['has_owner'])
                self.assertNotIn(('status', True), self.engine.calls)
        self.engine.active = False
        self.start()
        status, _, body = self.request('POST', '/v1/stop', {}, auth=True)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['lease']['has_owner'])
        self.assertIsNone(self.server.owner_client_id)

    def test_failed_resume_does_not_claim_a_finished_session(self):
        self.engine.state = 'paused'
        status, _, _ = self.request('POST', '/v1/resume', {'client_id': OBSERVER}, auth=True)
        self.assertEqual(status, 409)
        self.assertIsNone(self.server.owner_client_id)

    def test_other_read_endpoints_never_refresh_owner_lease(self):
        self.start()
        self.engine.calls.clear()
        for path in ('/v1/windows', '/v1/frame', '/v1/health'):
            status, _, _ = self.request('GET', path, auth=True, headers={CLIENT_HEADER: OWNER})
            self.assertEqual(status, 200)
        self.assertNotIn(('status', True), self.engine.calls)

    def test_concurrent_starts_preserve_winning_page_ownership(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(self.start, (OWNER, OBSERVER)))
        self.assertEqual(sorted(item[0] for item in results), [200, 409])
        winner = (OWNER, OBSERVER)[next(i for i, item in enumerate(results) if item[0] == 200)]
        self.assertEqual(self.server.owner_client_id, winner)

    def test_closed_owner_expires_even_while_other_tabs_poll(self):
        class TimedFakeEngine(support.FakeEngine):
            def __init__(self):
                super().__init__()
                self.now = 0
                self.last_lease = 0

            def status(self, *, heartbeat=False):
                if heartbeat:
                    self.last_lease = self.now
                return super().status(heartbeat=heartbeat)

            def advance_to(self, now):
                self.now = now
                if self.active and now - self.last_lease > 20:
                    self.state = 'paused'

        self.engine = self.server.engine = TimedFakeEngine()
        self.start()
        self.status(OWNER)
        for second in (5, 10, 15, 19, 21):
            self.engine.advance_to(second)
            self.status(OBSERVER)
            self.status()  # A still-open legacy tab must also stay read-only.
        self.assertEqual(self.engine.state, 'paused')
        self.assertEqual(self.engine.last_lease, 0)
        self.assertEqual(self.engine.calls.count(('status', True)), 1)
        self.assertIsNone(self.server.owner_client_id)


if __name__ == '__main__':
    unittest.main()
