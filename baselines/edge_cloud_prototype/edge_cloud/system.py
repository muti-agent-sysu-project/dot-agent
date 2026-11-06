import json
import time
import uuid
from dataclasses import asdict

from .client import Call, ChatClient
from .config import Config

MODES = ('local', 'cloud', 'hybrid', 'dot')
STAGES = ('decompose', 'execute', 'aggregate', 'fallback')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def usage_metrics(calls, role=None):
    selected = [call for call in calls if call['attempted'] and
                (role is None or call['role'] == role)]
    known = sum(call['total_tokens'] for call in selected
                if call['total_tokens'] is not None)
    unknown = sum(call['total_tokens'] is None for call in selected)
    return {'cloud_tokens' if role == 'cloud' else 'tokens': None if unknown else known,
            'known_tokens_subtotal': known, 'unknown_usage_calls': unknown,
            'attempts': len(selected),
            'elapsed_ms': sum(call['elapsed_ms'] for call in selected)}


class System:
    def __init__(self, config: Config):
        self.config = config
        self.clients = {'local': ChatClient(config.local),
                        'cloud': ChatClient(config.cloud) if config.cloud else None}

    def route(self, prompt):
        if len(prompt) >= self.config.long_prompt_chars:
            return 'cloud', 'long_prompt'
        if any(keyword.casefold() in prompt.casefold()
               for keyword in self.config.complex_keywords):
            return 'cloud', 'complex_keyword'
        return 'local', 'default_local'

    def ask(self, prompt, mode='hybrid'):
        start = time.perf_counter()
        if mode not in MODES:
            raise ValueError('未知模式')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('输入不可为空')
        request_id = str(uuid.uuid4())
        calls = []
        events = []
        subtasks = []
        upgrade_count = 0
        degradation_count = 0
        error = None

        def execute(role, user_prompt, stage, subtask_id=None, max_tokens=None,
                    response_format=None):
            client = self.clients[role]
            call = (client.complete(user_prompt, self.config.system_prompt,
                                    max_tokens or self.config.max_tokens,
                                    self.config.temperature, response_format=response_format)
                    if client else Call('', error='not_configured'))
            record = asdict(call)
            record.pop('answer')
            endpoint = self.config.local if role == 'local' else self.config.cloud
            calls.append(dict(record, role=role, request_id=request_id,
                              call_id=str(uuid.uuid4()), stage=stage,
                              subtask_id=subtask_id,
                              requested_model=endpoint.model if endpoint else None))
            return call

        def within_budget(user_prompt):
            size = len(self.config.system_prompt.encode('utf-8')) + len(user_prompt.encode('utf-8'))
            if size <= self.config.dot_context_max_bytes:
                return True
            events.append('context_budget_exceeded')
            return False

        def fallback():
            nonlocal degradation_count, error
            degradation_count += 1
            if not within_budget(prompt):
                error = 'context_budget_exceeded'
                return Call('', error=error)
            result = execute('cloud', prompt, 'fallback')
            if not result.success:
                error = result.error
            return result

        if mode == 'dot':
            reason = 'dot'
            planner_prompt = (f'将原问题分解为1到{self.config.dot_max_subtasks}个顺序子任务。'
                              '只输出严格JSON对象，格式为'
                              '{"subtasks":["任务1"]}；仅写任务，不写推理过程。原问题：' + prompt)
            if not within_budget(planner_prompt):
                result = fallback()
            else:
                plan = execute('cloud', planner_prompt, 'decompose',
                               max_tokens=self.config.dot_planner_max_tokens,
                               response_format={'type': 'json_object'})
                if not plan.success:
                    result = plan
                    error = plan.error
                else:
                    try:
                        parsed = json.loads(plan.answer, object_pairs_hook=unique_object)
                        values = parsed['subtasks'] if isinstance(parsed, dict) and set(parsed) == {'subtasks'} else None
                        if (not isinstance(values, list) or
                                not 1 <= len(values) <= self.config.dot_max_subtasks or
                                not all(isinstance(item, str) and item.strip() for item in values)):
                            raise ValueError('invalid plan')
                    except (ValueError, TypeError, KeyError):
                        events.append('invalid_decomposition')
                        result = fallback()
                    else:
                        result = plan
                        completed = []
                        for index, task in enumerate(values, 1):
                            role, route_reason = self.route(task)
                            item = {'subtask_id': index, 'task': task,
                                    'role': role, 'assigned_role': role,
                                    'executed_role': None, 'route_reason': route_reason,
                                    'answer': '', 'success': False}
                            subtasks.append(item)
                            user_prompt = ('原问题：' + prompt + '\n当前子任务：' + task +
                                           '\n前置结果：' + json.dumps(completed, ensure_ascii=False))
                            if not within_budget(user_prompt):
                                result = fallback()
                                break
                            result = execute(role, user_prompt, 'execute', index)
                            item['executed_role'] = role
                            if not result.success and role == 'local':
                                upgrade_count += 1
                                events.append('local_upgraded_to_cloud')
                                item['role'] = 'cloud'
                                result = execute('cloud', user_prompt, 'execute', index)
                                item['executed_role'] = 'cloud'
                            item['success'] = result.success
                            item['answer'] = result.answer if result.success else ''
                            if not result.success:
                                error = result.error
                                break
                            completed.append({'subtask_id': index, 'task': task,
                                              'answer': result.answer})
                        else:
                            aggregate_prompt = ('原问题：' + prompt + '\n子任务结果：' +
                                                json.dumps(completed, ensure_ascii=False) +
                                                '\n请根据以上结果给出原问题的最终答案。')
                            if within_budget(aggregate_prompt):
                                result = execute('cloud', aggregate_prompt, 'aggregate')
                                if not result.success:
                                    error = result.error
                            else:
                                result = fallback()
        else:
            target, reason = self.route(prompt) if mode == 'hybrid' else (mode, 'baseline')
            result = execute(target, prompt, 'execute')
            if (mode == 'hybrid' and target == 'local' and not result.success
                    and self.config.fallback_on_local_error):
                upgrade_count += 1
                result = execute('cloud', prompt, 'fallback')
            if not result.success:
                error = result.error

        cloud = usage_metrics(calls, 'cloud')
        local = usage_metrics(calls, 'local')
        stage_metrics = {}
        for stage in STAGES:
            stage_calls = [call for call in calls if call['stage'] == stage]
            metrics = usage_metrics(stage_calls, 'cloud')
            local_stage = usage_metrics(stage_calls, 'local')
            metrics.update(local_tokens=local_stage['tokens'],
                           known_local_tokens_subtotal=local_stage['known_tokens_subtotal'],
                           local_unknown_usage_calls=local_stage['unknown_usage_calls'],
                           local_attempts=local_stage['attempts'],
                           all_call_elapsed_ms=sum(call['elapsed_ms'] for call in stage_calls))
            stage_metrics[stage] = metrics
        return {'mode': mode, 'request_id': request_id,
                'path': [call['role'] for call in calls], 'route_reason': reason,
                'success': result.success, 'answer': result.answer, 'model': result.model,
                'cloud_tokens': cloud['cloud_tokens'],
                'known_cloud_tokens_subtotal': cloud['known_tokens_subtotal'],
                'unknown_usage_calls': cloud['unknown_usage_calls'],
                'local_tokens': local['tokens'],
                'stage_metrics': stage_metrics, 'subtasks': subtasks,
                'events': events, 'upgrade_count': upgrade_count,
                'degradation_count': degradation_count, 'error': error,
                'elapsed_ms': (time.perf_counter() - start) * 1000,
                'calls': calls, 'simulated': False}
