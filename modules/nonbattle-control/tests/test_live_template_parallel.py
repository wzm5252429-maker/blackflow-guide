"""Exact template search batching stays ordered, bounded and fail-closed."""
import threading
import unittest
from unittest.mock import patch

import numpy as np

from blackflow_live.vision import MaaTemplates, TemplateHit, _match_template_batch


class LiveTemplateParallelTests(unittest.TestCase):
    def test_parallel_color_matches_equal_serial_results(self):
        rng = np.random.default_rng(47)
        image = rng.integers(0, 256, (100, 160, 3), dtype=np.uint8)
        templates = object.__new__(MaaTemplates)
        templates._cache = {}
        requests = []
        for index, (x, y) in enumerate(((5, 8), (45, 20), (83, 50), (115, 70))):
            name = f'template-{index}'
            templates._cache[name] = image[y:y+13, x:x+17].copy()
            requests.append((name, {'threshold': .95, 'maximum': 2, 'scales': (.8, 1.0, 1.2)}))
        before = image.copy()
        serial = [templates.match(image, name, **options) for name, options in requests]
        parallel = templates.match_many(image, requests)
        self.assertEqual(parallel, serial)
        self.assertTrue(all(len(hits) == 1 and hits[0].confidence >= .99 for hits in parallel))
        np.testing.assert_array_equal(image, before)

    def test_workers_are_bounded_and_results_keep_request_order(self):
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=3)
        active = peak = 0
        loaded = []
        caller = threading.get_ident()
        templates = object.__new__(MaaTemplates)

        def preload(name):
            self.assertEqual(threading.get_ident(), caller)
            loaded.append(name)

        def match(image, name, **options):
            nonlocal active, peak
            self.assertEqual(len(loaded), 9)
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if int(name) < 4:
                    barrier.wait()
                return [TemplateHit(name, .99, (int(name), 0, 10, 10), 1)]
            finally:
                with lock:
                    active -= 1

        templates.template, templates.match = preload, match
        result = templates.match_many(np.zeros((1, 1, 3), np.uint8), [(str(i), {}) for i in range(9)])
        self.assertEqual([hits[0].name for hits in result], [str(i) for i in range(9)])
        self.assertEqual(peak, 4)
        self.assertEqual(active, 0)
        self.assertFalse(any(t.name.startswith('blackflow-template') for t in threading.enumerate()))

    def test_worker_failure_is_not_reported_as_absent_battle_marker(self):
        templates = object.__new__(MaaTemplates)
        templates.template = lambda _: None

        def match(image, name, **options):
            if name == 'battle':
                raise RuntimeError('template search failed')
            return []

        templates.match = match
        with self.assertRaisesRegex(RuntimeError, 'template search failed'):
            templates.match_many(None, [('map', {}), ('battle', {}), ('ending', {})])
        self.assertFalse(any(t.name.startswith('blackflow-template') for t in threading.enumerate()))

    def test_empty_and_single_batches_do_not_create_threads(self):
        templates = object.__new__(MaaTemplates)
        templates.template = lambda _: None
        templates.match = lambda image, name, **options: [name, options]
        with patch('blackflow_live.vision.ThreadPoolExecutor', side_effect=AssertionError('unnecessary worker')):
            self.assertEqual(templates.match_many(None, []), [])
            self.assertEqual(templates.match_many(None, [('map', {'maximum': 2})]), [['map', {'maximum': 2}]])

    def test_legacy_template_provider_keeps_serial_compatibility(self):
        class Templates:
            def match(self, image, name, **options):
                return [name, options]

        self.assertEqual(_match_template_batch(Templates(), None, [('map', {'threshold': .88})]),
                         [['map', {'threshold': .88}]])


if __name__ == '__main__':
    unittest.main()
