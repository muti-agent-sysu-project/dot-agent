import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from edge_cloud.client import ChatClient
from edge_cloud.__main__ import DemoSystem
from edge_cloud.config import Config, Endpoint
from edge_cloud.evaluation import summarize, score, load_tasks, evaluate
from edge_cloud.system import System


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).calls.append((self.path, body, self.headers.get('Authorization')))
        model = body['model']
        if model == 'failed':
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'private-provider-error-secret')
            return
        if model == 'redirect':
            self.send_response(307)
            self.send_header('Location', '/other')
            self.end_headers()
            return
        if model == 'bad-json':
            data = b'not-json'
        else:
            response = {'model': model, 'choices': [{'message': {'content': '42'},
                                                   'finish_reason': 'stop'}]}
            if model != 'no-usage':
                response['usage'] = {'prompt_tokens': 10, 'completion_tokens': 2,
                                     'total_tokens': 12}
            if model == 'truncated':
                response['choices'][0]['finish_reason'] = 'length'
            if model == 'empty':
                response['choices'][0]['message']['content'] = ''
            data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(data)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.calls.clear()

    def system(self, local='local', cloud='cloud'):
        return System(Config(Endpoint(self.url, local),
                             Endpoint(self.url, cloud) if cloud else None))

    def test_protocol_chinese_and_auth(self):
        with patch.dict(os.environ, {'TEST_MODEL_KEY': 'private-test-key'}):
            client = ChatClient(Endpoint(self.url, 'local', 'TEST_MODEL_KEY'))
            result = client.complete('中文问题', 'system', 30, 0)
        self.assertEqual(result.answer, '42')
        path, body, auth = Handler.calls[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(body['messages'][1]['content'], '中文问题')
        self.assertEqual(body['max_tokens'], 30)
        self.assertFalse(body['stream'])
        self.assertEqual(auth, 'Bearer private-test-key')

    def test_three_modes_and_cloud_cost(self):
        system = self.system()
        local = system.ask('简单问题', 'local')
        cloud = system.ask('简单问题', 'cloud')
        hybrid = system.ask('简单问题', 'hybrid')
        self.assertEqual(local['cloud_tokens'], 0)
        self.assertEqual(cloud['cloud_tokens'], 12)
        self.assertEqual(hybrid['path'], ['local'])
        self.assertGreaterEqual(hybrid['elapsed_ms'], hybrid['calls'][0]['elapsed_ms'])

    def test_complex_direct_to_cloud(self):
        result = self.system().ask('请证明这个结论', 'hybrid')
        self.assertEqual(result['path'], ['cloud'])
        self.assertEqual(len(Handler.calls), 1)

    def test_local_failure_upgrades_once(self):
        result = self.system(local='failed').ask('简单问题', 'hybrid')
        self.assertTrue(result['success'])
        self.assertEqual(result['path'], ['local', 'cloud'])
        self.assertEqual(result['cloud_tokens'], 12)
        self.assertEqual(len(Handler.calls), 2)
        self.assertNotIn('private-provider', json.dumps(result))

    def test_missing_cloud_no_fabricated_response(self):
        result = self.system(cloud=None).ask('请证明结论', 'hybrid')
        self.assertFalse(result['success'])
        self.assertEqual(result['calls'][0]['error'], 'not_configured')
        self.assertEqual(Handler.calls, [])
        self.assertEqual(result['cloud_tokens'], 0)

    def test_failed_cloud_unknown_cost_and_no_retry(self):
        result = self.system(cloud='failed').ask('问题', 'cloud')
        self.assertFalse(result['success'])
        self.assertIsNone(result['cloud_tokens'])
        self.assertEqual(len(Handler.calls), 1)
        self.assertEqual(result['calls'][0]['error'], 'http_500')

    def test_missing_usage_is_unknown(self):
        result = self.system(cloud='no-usage').ask('问题', 'cloud')
        self.assertTrue(result['success'])
        self.assertIsNone(result['cloud_tokens'])
        summary = summarize([dict(result, correct=True)])['cloud']
        self.assertIsNone(summary['cloud_tokens'])
        self.assertEqual(summary['unknown_usage_calls'], 1)

    def test_redirect_not_followed(self):
        result = self.system(cloud='redirect').ask('问题', 'cloud')
        self.assertFalse(result['success'])
        self.assertEqual(result['calls'][0]['error'], 'http_307')
        self.assertEqual(len(Handler.calls), 1)

    def test_bad_json_and_empty(self):
        for model in ('bad-json', 'empty'):
            result = self.system(cloud=model).ask('问题', 'cloud')
            self.assertFalse(result['success'])
            self.assertEqual(result['cloud_tokens'], None if model == 'bad-json' else 12)

    def test_truncated_not_counted_as_success_but_usage_counted(self):
        result = self.system(cloud='truncated').ask('问题', 'cloud')
        self.assertFalse(result['success'])
        self.assertEqual(result['cloud_tokens'], 12)

    def test_missing_key_stops_before_request(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(Endpoint(self.url, 'local'), Endpoint(self.url, 'cloud', 'KEY'))
            result = System(cfg).ask('问题', 'cloud')
        self.assertFalse(result['success'])
        self.assertEqual(Handler.calls, [])
        self.assertEqual(result['cloud_tokens'], 0)

    def test_scoring_requires_whole_answer(self):
        task = {'answers': ['42']}
        self.assertTrue(score('42。', task))
        self.assertFalse(score('不是42，而是43', task))
        self.assertFalse(score('142', task))

    def test_invalid_mode_and_blank_input(self):
        with self.assertRaises(ValueError):
            self.system().ask('hi', 'invalid')
        with self.assertRaises(ValueError):
            self.system().ask('  ', 'local')

    def test_invalid_endpoint_and_generation_settings(self):
        with self.assertRaises(ValueError):
            Endpoint('https://user:secret@example.com/v1', 'model')
        with self.assertRaises(ValueError):
            Config(Endpoint(self.url, 'model'), max_tokens=0)

    def test_duplicate_tasks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.jsonl'
            task = {'id': 'a', 'prompt': 'question', 'answers': ['42']}
            path.write_text((json.dumps(task) + '\n') * 2, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_tasks(path)

    def test_demo_cannot_be_presented_as_real_zero_cost(self):
        system = DemoSystem(Config(Endpoint(self.url, 'local')))
        row = system.ask('question', 'cloud')
        self.assertTrue(row['simulated'])
        self.assertIsNone(summarize([row])['cloud']['cloud_tokens'])

    def test_evaluation_records_failures_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'tasks.jsonl'
            dataset.write_text(json.dumps({'id': 'a', 'prompt': '问题', 'answers': ['42']}) + '\n', encoding='utf-8')
            report = evaluate(self.system(cloud='failed'), dataset, ['local', 'cloud'], 2, root / 'out')
            self.assertEqual(report['results']['local']['accuracy'], 1)
            self.assertEqual(report['results']['cloud']['accuracy'], 0)
            self.assertEqual(report['results']['cloud']['errors'], 2)
            self.assertIsNone(report['results']['cloud']['cloud_tokens'])
            self.assertEqual(len((root / 'out/requests.jsonl').read_text().splitlines()), 4)
            count = len(Handler.calls)
            with self.assertRaises(FileExistsError):
                evaluate(self.system(), dataset, ['local'], 1, root / 'out')
            self.assertEqual(len(Handler.calls), count)


if __name__ == '__main__':
    unittest.main()
