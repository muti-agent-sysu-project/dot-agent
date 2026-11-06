"""Offline HTTP contract tests; all credentials below are fake."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import tempfile
import runpy
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dot_agent.gateway.factory import make_clients, warmup_real
from dot_agent.gateway.openai_compat import OpenAICompatClient, GenerationError
from dot_agent.gateway.meter import CallMeter
from dot_agent.gateway.metered import MeteredClient
from dot_agent.types import ModelTier
from dot_agent.pipeline import AgentPipeline


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        self.send_response(self.server.status)
        if self.server.status == 302:
            self.send_header('Location', '/must-not-follow')
        self.end_headers()
        self.wfile.write(json.dumps(self.server.body).encode())


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.status = 200
        self.server.body = {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 7, 'completion_tokens': 2}}
        self.server.seen = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.client = OpenAICompatClient(self.url, 'fake-test-key', 'test', thinking='disabled')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_cloud_payload_and_usage(self):
        result = asyncio.run(self.client.generate('hello'))
        path, headers, body = self.server.seen[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(headers['Authorization'], 'Bearer fake-test-key')
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (7, 2))
        self.assertFalse(result.usage_missing)

    def test_local_factory_never_inherits_cloud_key(self):
        with patch.dict(os.environ, {'LOCAL_BASE_URL': self.url, 'LOCAL_MODEL': 'qwen-local',
                        'CLOUD_API_KEY': 'fake-test-key', 'CLOUD_THINKING': 'disabled',
                        'DOT_REQUIRE_REAL': '1'}, clear=True):
            local, cloud, _ = make_clients()
            asyncio.run(local.generate('hello'))
            self.assertIs(local.tier, ModelTier.MD)
            self.assertNotIn('Authorization', self.server.seen[0][1])
            self.assertNotIn('thinking', self.server.seen[0][2])
            self.assertEqual(cloud.thinking, 'disabled')
            self.assertIsNone(AgentPipeline(local, cloud).cloud.fallback)

    def test_strict_refuses_stub_but_demo_still_works(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(make_clients()[2]['local'], 'stub')
        with patch.dict(os.environ, {'DOT_REQUIRE_REAL': '1'}, clear=True):
            with self.assertRaises(RuntimeError):
                make_clients()

    def test_bad_answers_keep_billable_usage(self):
        for content, finish in [('partial', 'length'), ('', 'stop'), (None, 'stop')]:
            with self.subTest(content=content, finish=finish):
                self.server.body['choices'][0] = {'message': {'content': content}, 'finish_reason': finish}
                meter = CallMeter()
                with patch('dot_agent.gateway.metered.store.record_call') as record:
                    with self.assertRaises(GenerationError):
                        asyncio.run(MeteredClient(self.client, meter, retries=0).generate('hi'))
                self.assertEqual(meter.cloud_in_tokens + meter.cloud_out_tokens, 9)
                self.assertGreater(meter.logs[0].latency_s, 0)
                self.assertEqual(record.call_args.kwargs['completion_tokens'], 2)

    def test_invalid_usage_is_unknown(self):
        for value in [None, -1, True, '2']:
            self.server.body['usage']['completion_tokens'] = value
            r = asyncio.run(self.client.generate('hi'))
            self.assertTrue(r.usage_missing)
            self.assertEqual((r.prompt_tokens, r.completion_tokens), (0, 0))

    def test_errors_redacted_and_redirect_refused(self):
        for status in [403, 302]:
            self.server.status = status
            self.server.body = {'error': 'fake-test-key private-provider-body'}
            self.server.seen.clear()
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(self.client.generate('hi'))
            self.assertEqual(str(ctx.exception), f'model_http_{status}')
            self.assertEqual(len(self.server.seen), 1)

    def test_fallback_tracks_requested_tier(self):
        self.server.body['choices'][0]['finish_reason'] = 'length'
        meter = CallMeter()
        from dot_agent.gateway.stub import StubModelClient
        local = MeteredClient(StubModelClient(ModelTier.MD, 'test-local'), meter, retries=0)
        cloud = MeteredClient(self.client, meter, retries=0, fallback=local)
        with patch('dot_agent.gateway.metered.store.record_call'):
            asyncio.run(cloud.generate('hi'))
        self.assertIs(meter.logs[-1].tier, ModelTier.MD)
        self.assertIs(meter.logs[-1].requested_tier, ModelTier.MC)
        self.assertEqual(meter.summarize()['cloud_fallbacks'], 1)

    def test_mainline_four_modes_use_gateway(self):
        from eval.runner import ModeRunner
        local = OpenAICompatClient(self.url, '', 'local', tier=ModelTier.MD)
        with patch.dict(os.environ, {'DOT_REQUIRE_REAL': '1'}), \
             patch('dot_agent.gateway.metered.store.record_call'), \
             patch('dot_agent.gateway.store.record_task'):
            for mode in ('local_only', 'cloud_only', 'rule', 'signal'):
                with self.subTest(mode=mode):
                    result = asyncio.run(ModeRunner(mode, local, self.client, 'offline').run(
                        {'query': 'What is AI?', 'score': 'contains', 'expected': 'OK'}))
                    self.assertTrue(result['correct'])
                    self.assertEqual(result['unknown'], 0)
        self.assertGreaterEqual(len(self.server.seen), 4)

    def test_secret_loader_accepts_only_valid_file(self):
        root = Path(__file__).resolve().parents[1]
        load = runpy.run_path(str(root / 'scripts/run_autodl.py'))['load_key']
        with tempfile.TemporaryDirectory(dir=root) as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / 'fake-secret'
            path.write_text('fake-test-key', encoding='utf-8')
            path.chmod(0o600)
            load(path)
            self.assertEqual(os.environ['CLOUD_API_KEY'], 'fake-test-key')
            for invalid in ('', 'contains whitespace', 'x' * 8193):
                path.write_text(invalid, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load(path)
            with self.assertRaises(ValueError):
                load(Path(tmp))

    def test_strict_preflight_requires_generation_usage(self):
        self.server.body['usage'] = {}
        with patch.dict(os.environ, {'DOT_REQUIRE_REAL': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'preflight_missing_usage'):
                asyncio.run(warmup_real({}, self.client, self.client))


if __name__ == '__main__':
    unittest.main()
