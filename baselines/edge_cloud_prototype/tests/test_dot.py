import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from edge_cloud.__main__ import DemoSystem
from edge_cloud.config import Config, Endpoint
from edge_cloud.evaluation import evaluate, summarize
from edge_cloud.system import System


class DoTHandler(BaseHTTPRequestHandler):
    requests = []
    queued = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).requests.append(body)
        response = type(self).queued.pop(0)
        if isinstance(response, int):
            self.send_response(response)
            self.end_headers()
            return
        content, usage = response
        data = {'model': body['model'] + '-response',
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        if usage is not None:
            data['usage'] = {'prompt_tokens': usage, 'completion_tokens': 2,
                             'total_tokens': usage + 2}
        encoded = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(encoded)


class DoTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), DoTHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        DoTHandler.requests.clear()
        DoTHandler.queued.clear()

    def system(self, **options):
        config = Config(Endpoint(self.url, 'local'),
                        Endpoint(self.url, 'cloud', thinking='disabled'), **options)
        return System(config)

    def test_plan_execute_in_order_and_aggregate_with_ids(self):
        DoTHandler.queued[:] = [
            ('{"subtasks":["求 1+1","证明 2+2=4"]}', 3),
            ('2', 4), ('4', 5), ('最终答案', 6)]
        row = self.system().ask('先计算再证明', 'dot')
        self.assertTrue(row['success'])
        self.assertEqual(row['answer'], '最终答案')
        self.assertEqual([c['stage'] for c in row['calls']],
                         ['decompose', 'execute', 'execute', 'aggregate'])
        self.assertEqual([c['role'] for c in row['calls']],
                         ['cloud', 'local', 'cloud', 'cloud'])
        self.assertEqual(row['cloud_tokens'], 20)
        self.assertEqual(row['known_cloud_tokens_subtotal'], 20)
        self.assertEqual([s['subtask_id'] for s in row['subtasks']], [1, 2])
        self.assertEqual(row['subtasks'][1]['answer'], '4')
        self.assertIn('先计算再证明', DoTHandler.requests[1]['messages'][1]['content'])
        self.assertIn('2', DoTHandler.requests[2]['messages'][1]['content'])
        self.assertEqual(DoTHandler.requests[0]['response_format'], {'type': 'json_object'})
        self.assertEqual(DoTHandler.requests[0]['max_tokens'], 512)
        self.assertEqual(DoTHandler.requests[0]['thinking'], {'type': 'disabled'})
        self.assertNotIn('thinking', DoTHandler.requests[1])
        self.assertTrue(all(c['request_id'] == row['request_id'] for c in row['calls']))
        self.assertEqual(len({c['call_id'] for c in row['calls']}), 4)
        self.assertEqual(row['calls'][0]['requested_model'], 'cloud')
        self.assertEqual(row['calls'][0]['model'], 'cloud-response')
        self.assertEqual(row['stage_metrics']['decompose']['cloud_tokens'], 5)
        execute_calls = [call for call in row['calls'] if call['stage'] == 'execute']
        self.assertEqual(row['stage_metrics']['execute']['local_tokens'], 6)
        self.assertEqual(row['stage_metrics']['execute']['all_call_elapsed_ms'],
                         sum(call['elapsed_ms'] for call in execute_calls))
        summary = summarize([row])['dot']['stage_metrics']['execute']
        self.assertEqual(summary['local_tokens'], 6)
        self.assertEqual(summary['all_call_elapsed_ms'],
                         sum(call['elapsed_ms'] for call in execute_calls))

    def test_invalid_plan_degrades_once_and_boundaries(self):
        invalid = ['{"subtasks":[]}', '{"subtasks":[" "]}',
                   '{"subtasks":[1]}', '{"subtasks":["a","b","c","d","e"]}',
                   'not-json', '{"subtasks":["a"],"extra":1}',
                   '{"subtasks":["a"],"subtasks":["b"]}']
        for plan in invalid:
            with self.subTest(plan=plan):
                DoTHandler.requests.clear()
                DoTHandler.queued[:] = [(plan, 3), ('退化答案', 4)]
                row = self.system().ask('原问题', 'dot')
                self.assertTrue(row['success'])
                self.assertEqual([c['stage'] for c in row['calls']], ['decompose', 'fallback'])
                self.assertEqual(row['degradation_count'], 1)
                self.assertEqual(summarize([row])['dot']['fallback_rate'], 1)
                self.assertEqual(len(DoTHandler.requests), 2)
                self.assertEqual(DoTHandler.requests[1]['messages'][1]['content'], '原问题')
        for count in (1, 4):
            with self.subTest(count=count):
                DoTHandler.requests.clear()
                DoTHandler.queued[:] = [(json.dumps({'subtasks': ['a'] * count}), 3)] + \
                    [('执行', 2)] * count + [('汇总', 4)]
                row = self.system().ask('原问题', 'dot')
                self.assertTrue(row['success'])
                self.assertEqual(len(row['subtasks']), count)

    def test_decomposition_http_failure_stops_without_retry(self):
        DoTHandler.queued[:] = [500]
        row = self.system().ask('问题', 'dot')
        self.assertFalse(row['success'])
        self.assertEqual(len(DoTHandler.requests), 1)
        self.assertEqual(row['calls'][0]['error'], 'http_500')
        self.assertEqual(row['degradation_count'], 0)

    def test_local_failure_upgrades_once_and_cloud_failure_stops(self):
        DoTHandler.queued[:] = [('{"subtasks":["第一步","第二步"]}', 3),
                               500, ('升级结果', None), 500, 500]
        row = self.system().ask('问题', 'dot')
        self.assertFalse(row['success'])
        self.assertEqual([c['role'] for c in row['calls']],
                         ['cloud', 'local', 'cloud', 'local', 'cloud'])
        self.assertEqual(row['upgrade_count'], 2)
        self.assertEqual(summarize([row])['dot']['fallback_rate'], 1)
        self.assertEqual(row['degradation_count'], 0)
        self.assertEqual(len(DoTHandler.requests), 5)
        self.assertIsNone(row['cloud_tokens'])
        self.assertEqual(row['known_cloud_tokens_subtotal'], 5)
        self.assertEqual([item['assigned_role'] for item in row['subtasks']],
                         ['local', 'local'])
        self.assertEqual([item['executed_role'] for item in row['subtasks']],
                         ['cloud', 'cloud'])
        self.assertEqual([item['role'] for item in row['subtasks']],
                         ['cloud', 'cloud'])
        self.assertEqual(row['subtasks'][0]['route_reason'], 'default_local')
        self.assertEqual(row['stage_metrics']['execute']['all_call_elapsed_ms'],
                         sum(call['elapsed_ms'] for call in row['calls']
                             if call['stage'] == 'execute'))

    def test_budget_exceeded_fallback_and_no_external_call_if_original_too_large(self):
        DoTHandler.queued[:] = [(json.dumps({'subtasks': ['长' * 100]}, ensure_ascii=False), 3),
                               ('退化', 4)]
        row = self.system(dot_context_max_bytes=250).ask('短问题', 'dot')
        self.assertTrue(row['success'])
        self.assertEqual([c['stage'] for c in row['calls']], ['decompose', 'fallback'])
        self.assertIn('context_budget_exceeded', row['events'])
        self.assertEqual(row['degradation_count'], 1)
        DoTHandler.requests.clear()
        DoTHandler.queued.clear()
        row = self.system(dot_context_max_bytes=10).ask('很长的原问题', 'dot')
        self.assertFalse(row['success'])
        self.assertEqual(DoTHandler.requests, [])
        self.assertEqual(row['cloud_tokens'], 0)
        self.assertIn('context_budget_exceeded', row['events'])

    def test_missing_usage_and_evaluation_linkage(self):
        DoTHandler.queued[:] = [('{"subtasks":["计算"]}', None),
                                  ('结果', 2), ('42', 3)]
        row = self.system().ask('问题', 'dot')
        self.assertIsNone(row['cloud_tokens'])
        self.assertEqual(row['unknown_usage_calls'], 1)
        self.assertEqual(row['known_cloud_tokens_subtotal'], 5)
        self.assertIsNone(row['stage_metrics']['decompose']['cloud_tokens'])
        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / 'tasks.jsonl'
            tasks.write_text(json.dumps({'id': 't1', 'prompt': '问题', 'answers': ['42']}) + '\n')
            DoTHandler.queued[:] = [('{"subtasks":["计算"]}', 3),
                                      ('结果', 2), ('42', 3)]
            report = evaluate(self.system(), tasks, ['dot'], 1, Path(directory) / 'out')
            saved = json.loads((Path(directory) / 'out/requests.jsonl').read_text())
            self.assertEqual(saved['experiment_id'], saved['calls'][0]['experiment_id'])
            self.assertEqual(saved['calls'][0]['task_id'], 't1')
            self.assertEqual(saved['calls'][0]['repeat'], 0)
            self.assertEqual(report['results']['dot']['accuracy'], 1)
            self.assertEqual(report['results']['dot']['upgrade_count'], 0)
            self.assertFalse(report['includes_warmup'])

    def test_aggregate_cloud_error_has_no_retry(self):
        DoTHandler.queued[:] = [('{"subtasks":["计算"]}', 3), ('2', 2), 500]
        row = self.system().ask('问题', 'dot')
        self.assertFalse(row['success'])
        self.assertEqual([c['stage'] for c in row['calls']],
                         ['decompose', 'execute', 'aggregate'])
        self.assertEqual(row['error'], 'http_500')
        self.assertEqual(len(DoTHandler.requests), 3)

    def test_config_bounds_and_example(self):
        example = Config.load(Path(__file__).resolve().parents[1] /
                              'config.deepseek.example.json')
        self.assertEqual(example.cloud.thinking, 'disabled')
        self.assertEqual(example.dot_context_max_bytes, 9000)
        for value in (0, 5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.system(dot_max_subtasks=value)
        with self.assertRaises(ValueError):
            Endpoint(self.url, 'cloud', thinking='invalid')

    def test_planner_prompt_uses_configured_subtask_limit(self):
        DoTHandler.queued[:] = [('{"subtasks":["计算"]}', 3), ('2', 2), ('2', 3)]
        row = self.system(dot_max_subtasks=2).ask('原问题', 'dot')
        self.assertTrue(row['success'])
        self.assertIn('1到2个', DoTHandler.requests[0]['messages'][1]['content'])

    def test_demo_dot_is_simulated_with_unknown_cloud_usage(self):
        config = Config(Endpoint(self.url, 'local'), Endpoint(self.url, 'cloud'))
        row = DemoSystem(config).ask('演示问题', 'dot')
        self.assertTrue(row['success'])
        self.assertTrue(row['simulated'])
        self.assertEqual(len(row['subtasks']), 2)
        self.assertIsNone(row['cloud_tokens'])
        self.assertIsNone(summarize([row])['dot']['cloud_tokens'])
        self.assertEqual(DoTHandler.requests, [])


if __name__ == '__main__':
    unittest.main()
