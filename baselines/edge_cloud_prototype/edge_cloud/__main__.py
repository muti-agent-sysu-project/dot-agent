import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .client import Call
from .config import Config
from .evaluation import evaluate
from .system import MODES, System


def doctor(system):
    rows = {}
    for role, client in system.clients.items():
        if client is None:
            rows[role] = {'status': 'not_configured'}
            continue
        endpoint = client.endpoint
        key = os.environ.get(endpoint.api_key_env) if endpoint.api_key_env else None
        if endpoint.api_key_env and not key:
            rows[role] = {'status': 'missing_api_key'}
            continue
        request = Request(endpoint.base_url.rstrip('/') + '/models')
        if key:
            request.add_header('Authorization', 'Bearer ' + key)
        try:
            with client.opener.open(request, timeout=min(endpoint.timeout_seconds, 10)) as response:
                data = json.load(response)
            names = [item['id'] for item in data['data']]
            rows[role] = {'status': 'ready' if endpoint.model in names else 'model_not_listed',
                          'configured_model': endpoint.model, 'listed_models': names}
        except HTTPError as error:
            rows[role] = {'status': f'http_{error.code}'}
            error.close()
        except (OSError, URLError, ValueError, KeyError, TypeError):
            rows[role] = {'status': 'unavailable'}
    return rows


class DemoClient:
    def complete(self, *args, **kwargs):
        # 固定测试回复，不读取参考答案；不代表模型能力。
        answer = ('{"subtasks":["读取题目条件","整理最终答案"]}'
                  if kwargs.get('response_format') else '42')
        return Call('SIMULATED-NOT-A-MODEL', answer=answer, success=True,
                    attempted=False, elapsed_ms=0, finish_reason='stop')


class DemoSystem(System):
    def __init__(self, config):
        super().__init__(config)
        self.clients = {'local': DemoClient(), 'cloud': DemoClient()}

    def ask(self, *args, **kwargs):
        result = super().ask(*args, **kwargs)
        result['simulated'] = True
        result['cloud_tokens'] = None
        for metric in result['stage_metrics'].values():
            metric['cloud_tokens'] = None
        return result


def main():
    parser = argparse.ArgumentParser(description='端云协同初版；真实模型与模拟演示明确区分')
    parser.add_argument('--config', default='config.example.json')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='检查模型列表，不发起计费聊天请求')
    ask = commands.add_parser('ask')
    ask.add_argument('prompt')
    ask.add_argument('--mode', choices=MODES, default='hybrid')
    for name in ('evaluate', 'demo'):
        evaluation = commands.add_parser(name)
        evaluation.add_argument('--dataset', default='data/smoke.jsonl')
        evaluation.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
        evaluation.add_argument('--repeats', type=int, default=1)
        evaluation.add_argument('--output', required=True, help='必须为新的报告目录')
    args = parser.parse_args()
    try:
        config = Config.load(args.config)
        system = DemoSystem(config) if args.command == 'demo' else System(config)
        if args.command == 'doctor':
            result = doctor(system)
            code = 0 if all(row['status'] == 'ready' for row in result.values()) else 2
        elif args.command == 'ask':
            result = system.ask(args.prompt, args.mode)
            code = 0 if result['success'] else 2
        else:
            result = evaluate(system, args.dataset, args.modes, args.repeats, args.output)
            code = 0 if all(row['errors'] == 0 for row in result['results'].values()) else 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return code
    except (OSError, ValueError, TypeError, KeyError) as error:
        # 不输出配置文件、凭据、供应商正文或异常字符串。
        print(json.dumps({'error': type(error).__name__,
                          'message': '检查配置、题集与输出目录（目录不可已存在）。'}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
