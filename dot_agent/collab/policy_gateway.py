"""Per-episode OpenAI planner endpoint for self-hosted EC² diagnostic runs."""
import asyncio
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
from threading import Thread
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import uvicorn
from jsonschema import Draft202012Validator
from referencing import Registry as SchemaRegistry

from .trajectory_view import ALIAS, FIELD, RULES, SessionView
from dot_agent.collab.privacy_ledger import TrajectoryPrivacyLedger, TransactionalBoundary
from dot_agent.collab.types import Action, BudgetExhaustedError, Candidate, PolicyDeniedError, Scope, Usage


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def predictor(path):
    # Estimated advantage of cloud over edge for each observable step family.
    priors = {"planning": .8, "echo": .05, "arithmetic": .1, "tool_result": .65}
    if path:
        fitted = json.loads(Path(path).read_text(encoding="utf8"))
        if fitted["version"] != 1 or set(fitted["advantage"]) != priors.keys():
            raise ValueError("invalid calibration schema")
        for k,v in fitted["advantage"].items():
            if type(v) not in (int,float) or not math.isfinite(v) or not -1 <= v <= 1:
                raise ValueError("invalid calibration estimate")
            priors[k] = v
    return priors


def family(messages):
    last = messages[-1]
    content = last.get("content")
    if last["role"] == "tool":
        if isinstance(content,str) and content.strip().lower() in {"ok","done","success","true","false"}:
            return "echo"
        return "tool_result"
    if isinstance(content,str) and re.fullmatch(r"compare -?\d+ and -?\d+",content.strip(),re.I):
        return "arithmetic"
    return "planning"


def weakest(messages):
    """Only project when the user's complete instruction specifies the sufficient statistic."""
    last = messages[-1]
    question = last.get("content")
    if isinstance(question,str) and last["role"] == "user":
        match = re.fullmatch(r"Is (-?\d{1,18}) greater than (-?\d{1,18})\?",question.strip(),re.I)
        if match:
            a,b = map(int,match.groups())
            rules = [m for m in messages[:-1] if m["role"] in {"system","developer"}]
            return [*rules, {**last,"content":f"Answer using this locally verified comparison: {a>b}."}], "predicate"
    if last["role"] == "tool" and isinstance(question,str):
        prompt = next((m.get("content") for m in reversed(messages) if m["role"]=="user"),None)
        if not isinstance(prompt,str):
            return messages,"typed"
        try:
            result = json.loads(question)
        except ValueError:
            return messages,"typed"
        rows = result if isinstance(result,list) else [result]
        columns = re.fullmatch(r"(?:Return|Show|Extract) only (?:the )?([a-zA-Z_]\w*(?:\s*(?:,|and)\s*[a-zA-Z_]\w*)*) fields?\.?",prompt.strip(),re.I)
        if columns:
            keys = re.split(r"\s*(?:,|\band\b)\s*",columns[1])
            if all(isinstance(row,dict) and all(k in row for k in keys) for row in rows):
                projected = [{k:row[k] for k in keys} for row in rows]
                sufficient = projected if isinstance(result,list) else projected[0]
                return [*messages[:-1],{**last,"content":json.dumps(sufficient)}], "column"
        comparison = re.fullmatch(r"Is (?:the )?(\w+) (greater|less) than (?:the )?(\w+)\?", prompt.strip(), re.I)
        if comparison and isinstance(result,dict):
            left,right = result.get(comparison[1]),result.get(comparison[3])
            if type(left) in (float,int) and type(right) in (float,int):
                bit = left>right if comparison[2].lower()=="greater" else left<right
                return [*messages[:-1],{**last,"content":json.dumps({"comparison_result":bit})}], "predicate"
    return messages, "typed"


def echo_contract(messages):
    # Compile just this pure identity workflow after two verified local executions.
    if messages[-1]["role"] != "tool":
        return None
    user = next((m for m in reversed(messages) if m["role"] == "user"), {})
    value = messages[-1].get("content")
    if user.get("content") == "Return the tool result verbatim." and isinstance(value,str) and len(value)<=4096:
        return value


def read_contract(req):
    """Pure read action with a verbatim local binding and an exact tool schema."""
    last = req["messages"][-1]
    if last["role"] != "user" or not isinstance(last.get("content"),str):
        return None
    match = re.fullmatch(r"(?:Please\s+)?(?:read|open|inspect|show)\s+(?:the\s+file\s+)?[\x60'\"]?((?:/tmp_workspace/)?[\w./-]+\.[A-Za-z0-9]{1,10})[\x60'\"]?[.!]?",last["content"].strip(),re.I)
    if not match:
        return None
    target = PurePosixPath(match[1])
    if (target.is_absolute() and not target.is_relative_to("/tmp_workspace")) or any(
            part.startswith(".") or part=="gt" for part in target.parts):
        return None
    for tool in req.get("tools",[]):
        fn=tool["function"]
        params=fn.get("parameters",{})
        if fn["name"] in {"read","read_file"} and params.get("type")=="object":
            key=next((k for k in ("path","file_path") if params.get("properties",{}).get(k,{}).get("type")=="string"),None)
            if key and set(params.get("required",[])) <= {key} and req.get("tool_choice","auto")=="auto":
                return fn["name"], key, match[1]
    return None


def completion(model,text):
    return {"id":"chatcmpl-"+uuid4().hex,"object":"chat.completion","created":0,"model":model,
            "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]}


def serialize(response, stream):
    if not stream:
        return b"application/json", encode(response)
    common = {k:response[k] for k in ("id","created","model") if k in response}
    common["object"] = "chat.completion.chunk"
    events = []
    for i,choice in enumerate(response["choices"]):
        delta = dict(choice["message"])
        if "tool_calls" in delta:
            delta["tool_calls"] = [dict(call,index=j) for j,call in enumerate(delta["tool_calls"])]
        for data,reason in [(delta,None),({},choice.get("finish_reason","stop"))]:
            events.append({**common,"choices":[{"index":choice.get("index",i),"delta":data,"finish_reason":reason}]})
    if "usage" in response:
        events.append({**common,"choices":[],"usage":response["usage"]})
    return b"text/event-stream", b"".join(b"data: "+encode(e)+b"\n\n" for e in events)+b"data: [DONE]\n\n"


@contextmanager
def serve(*, arm, task_id, output, cloud_url, cloud_model, edge_model, edge_port,
          calibrated_predictor=None, flow=None, p_max=.6, c_max=100_000_000):
    if arm not in {"B6_paac","B8_modular","B9_joint"}:
        raise ValueError("invalid arm")
    url = urlsplit(cloud_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1","::1"} or url.path != "/v1" or url.username or url.password or url.query or url.fragment:
        raise ValueError("loopback recorder required")
    registry = SessionView()
    priors = predictor(calibrated_predictor)
    flow = {} if flow is None else flow
    metrics = Counter()
    reviewed = {}
    pending_reads = {}
    lock = asyncio.Lock()
    scope = Scope(task_id,"self-hosted-cloud","trajectory-classes-v1")
    kinds = set(FIELD.values()) | {k for k,_ in RULES} | {"PREDICATE","LINK"}
    ledger = TrajectoryPrivacyLedger(dict.fromkeys(kinds,1), rules=[({"PERSON","MONEY"},"LINK")])

    async def post(url, wire):
        async with httpx.AsyncClient(timeout=300,trust_env=False) as client:
            reply = await client.post(url+"/chat/completions",content=wire,headers={"Content-Type":"application/json"})
            reply.raise_for_status()
            return reply.content

    async def transport(scope,wire,cap,attempt):
        start = monotonic()
        body = await post(cloud_url,wire)
        u = json.loads(body).get("usage") or {}
        return body, Usage.from_inclusive_input(attempt,u.get("prompt_tokens"),
                        (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        u.get("completion_tokens"),"ok",monotonic()-start)

    def inspect(wire,action,cap):
        if (wire,action,cap) not in reviewed:
            raise PolicyDeniedError("unreviewed request")
        return reviewed[wire,action,cap]

    boundary = TransactionalBoundary(output/"policy.sqlite",scope,ledger,inspect=inspect,input_bound=len,
                                     transport=transport,p_max=p_max,c_max=c_max)

    async def handle(req):
        messages = req["messages"]
        kind = family(messages)
        identity = echo_contract(messages) if arm=="B9_joint" and len(messages)>=3 and not req.get("tool_choice") else None
        contract = sha256(encode([req.get("tools",[]), [m for m in messages if m["role"] in {"system","developer"}]])).digest()
        echo_key = ("verified_echo", contract)
        binding = read_contract(req) if arm=="B9_joint" else None
        # A tool response is the first local evidence that the proposed read ran.
        last = messages[-1]
        if last["role"]=="tool" and (previous := pending_reads.pop(last.get("tool_call_id"),None)):
            result = last.get("content")
            if (len(messages)>1 and messages[-2].get("role")=="assistant" and
                any(c.get("id")==last["tool_call_id"] and c.get("function",{}).get("name")==previous[0][1]
                    and json.loads(c["function"]["arguments"])==previous[1] for c in messages[-2].get("tool_calls",[]))
                and isinstance(result,str) and result.strip() and not re.search(r"error|failed|denied|not found",result,re.I)):
                flow[previous[0]] = flow.get(previous[0],0)+1
        workflow = ("verified_read", *binding[:2], contract) if binding else None
        if binding and flow.get(workflow,0)>=2:
            name,key,path = binding
            answer = completion(edge_model,None)
            choice = answer["choices"][0]
            choice["finish_reason"]="tool_calls"
            choice["message"]["tool_calls"]=[{"type":"function","id":"call_"+uuid4().hex,
                "function":{"name":name,"arguments":encode({key:path}).decode()}}]
            metrics["hot_hits"] += 1
            return answer

        async def edge():
            metrics["edge"] += 1
            wire = encode({**req,"model":edge_model,"stream":False})
            reply = json.loads(await post(f"http://127.0.0.1:{edge_port}/v1",wire))
            if identity is not None and reply["choices"]:
                msg = reply["choices"][0]["message"]
                if msg.get("content")==identity and not msg.get("tool_calls"):
                    flow[echo_key] = flow.get(echo_key,0)+1
            return reply

        if identity is not None and flow.get(echo_key,0)>=2:
            metrics["hot_hits"] += 1
            return completion(edge_model,identity)
        if arm!="B6_paac" and (identity is not None or priors[kind] <= .2):
            return await edge()

        view, mode = weakest(messages) if arm=="B9_joint" else (messages,"typed")
        cap = req.get("max_completion_tokens",req.get("max_tokens",4096))
        if type(cap) is not int or not 1<=cap<=131072:
            raise ValueError("invalid output cap")
        # Construct a new outbound envelope, dropping user identifiers and vendor metadata.
        try:
            masked_messages = registry.messages({"messages":view})
            policy_note = ("Private values use typed __EC2_ references. Copy a reference exactly when its value "
                           "is needed in text or a tool argument. Use a JSON string for the reference even "
                           "when the underlying parameter is numeric. The local executor restores its type. "
                           "Do not infer a hidden value. Use a supplied comparison result directly.")
            outgoing = {"model":cloud_model,"messages":[{"role":"system","content":policy_note},*masked_messages],
                        "max_tokens":cap,"stream":False}
            if "tools" in req:
                outgoing["tools"] = registry.tools(req["tools"])
            for key in ("temperature","top_p","parallel_tool_calls","seed","n"):
                if key in req and type(req[key]) in (int,float,bool):
                    outgoing[key]=req[key]
            if "response_format" in req:
                fmt = req["response_format"]
                if fmt == {"type":"json_object"} or fmt == {"type":"text"}:
                    outgoing["response_format"] = fmt
                else:
                    raise ValueError("unsupported structured output")
            if "tool_choice" in req:
                choice = req["tool_choice"]
                names = {t["function"]["name"] for t in outgoing.get("tools",[])}
                if isinstance(choice,str) and choice in {"auto","none","required"}:
                    outgoing["tool_choice"]=choice
                elif isinstance(choice,dict) and choice["function"]["name"] in names:
                    outgoing["tool_choice"]={"type":"function","function":{"name":choice["function"]["name"]}}
                else:
                    raise ValueError("unsupported tool choice")
            registry.check_envelope(outgoing)
            wire = encode(outgoing)
        except (KeyError,TypeError,ValueError):
            metrics["unsupported"] += 1
            if arm=="B6_paac":
                raise ValueError("unsupported PAAC envelope") from None
            return await edge()

        if arm=="B6_paac":
            metrics["cloud_attempts"] += 1
            body = await post(cloud_url,wire)
        else:
            facts = set()  # populated from final envelope only
            for token in ALIAS.findall(wire.decode()):
                if token not in registry.reverse:
                    raise ValueError("foreign alias")
                tag = token.strip("_").split("_")[2]
                if tag in kinds:
                    facts.add(tag)
            if mode=="predicate":
                facts.add("PREDICATE")
            candidate = Candidate(Action.DELEGATE,mode,wire,cap)
            reviewed[wire,candidate.action,cap]=facts
            try:
                if arm=="B8_modular":
                    delta = float(boundary.standalone_quote(candidate))
                    price = 0
                else:
                    delta = float(boundary.quote(candidate))
                    spent,_ = boundary.totals()
                    remaining = p_max-float(spent)
                    if req.get("tools") and mode=="typed" and delta>remaining*.75:
                        metrics["reserved_for_later"] += 1
                        return await edge()
                    price = flow.get("price",0) + delta/max(.001,remaining)
                cost = min(.35,len(wire)/200000) + .2 + price*delta
                reuse_value = .08*min(2,flow.get(workflow,0)+1) if binding else 0
                if arm=="B9_joint":
                    cost += flow.get("token_price",0)*min(1,(len(wire)+cap)/32768)
                if priors[kind] + reuse_value <= cost:
                    metrics["price_local"] += 1
                    return await edge()
                permit = boundary.reserve(scope,candidate,uuid4().hex)
            except (PolicyDeniedError,BudgetExhaustedError):
                metrics["budget_local"] += 1
                return await edge()
            finally:
                reviewed.clear()
            metrics["cloud_attempts"] += 1
            body,_ = await boundary.send(permit)

        metrics["view_"+mode] += 1
        reply = json.loads(body)
        for choice in reply["choices"]:
            msg = choice["message"]
            if isinstance(msg.get("content"),str):
                msg["content"]=str(registry.restore(msg["content"]))
            for call in msg.get("tool_calls") or []:
                fn = call["function"]
                declared = {t["function"]["name"]:t["function"] for t in req.get("tools",[])}
                if fn["name"] not in declared:
                    raise ValueError("unknown response tool")
                args = registry.restore(json.loads(fn["arguments"]),
                                        command=fn["name"].lower() in {"exec","bash","shell","python"})
                validator = Draft202012Validator(declared[fn["name"]].get("parameters",{"type":"object"}),registry=SchemaRegistry())
                if not isinstance(args,dict) or not validator.is_valid(args):
                    raise ValueError("restored arguments violate tool schema")
                fn["arguments"]=encode(args).decode()
                if binding and fn["name"]==binding[0] and args=={binding[1]:binding[2]}:
                    pending_reads[call["id"]]=(workflow,args)
        return reply

    async def app(scope,receive,send):
        status, mime, body = 404, b"application/json", b"{}"
        if scope["method"]=="POST" and scope["path"]=="/v1/chat/completions":
            try:
                data = bytearray()
                while True:
                    event = await receive()
                    if event["type"]!="http.request":
                        return
                    data.extend(event["body"])
                    if len(data)>8*1024*1024:
                        raise ValueError("request too large")
                    if not event.get("more_body"):
                        break
                req=json.loads(data)
                stream=req.pop("stream",False)
                req.pop("stream_options",None)
                async with lock:
                    if arm=="B9_joint":
                        before_p,before_c = boundary.totals()
                        try:
                            reply=await handle(req)
                        finally:
                            after_p,after_c = boundary.totals()
                            flow["token_price"] = min(1,max(0,flow.get("token_price",0)+.05*((after_c-before_c)/4096-1)))
                            flow["price"] = min(5,max(0,flow.get("price",0)+.25*(float(after_p-before_p)-p_max/16)))
                    else:
                        reply=await handle(req)
                mime,body=serialize(reply,stream)
                metrics["served_steps"] += 1
                status=200
            except (ValueError,TypeError,KeyError,IndexError) as exc:
                metrics["rejected_steps"] += 1
                causes = {"malformed alias", "unknown alias", "unsafe command substitution",
                          "restored arguments violate tool schema", "unknown response tool",
                          "registered literal in outbound envelope", "requires local multimodal processing",
                          "reserved marker in local input"}
                if str(exc) in causes:
                    metrics["rejected_" + str(exc).replace(" ", "_")] += 1
                status,body=422,b'{"error":"invalid or unsupported envelope"}'
            except Exception:
                metrics["failed_steps"] += 1
                status,body=502,b'{"error":"planner transport or contract failed"}'
        await send({"type":"http.response.start","status":status,"headers":[(b"content-type",mime)]})
        await send({"type":"http.response.body","body":body})

    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        server=uvicorn.Server(uvicorn.Config(app,lifespan="off",log_level="error",access_log=False))
        thread=Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])),daemon=True)
        thread.start()
        try:
            deadline=monotonic()+10
            while not server.started:
                if not thread.is_alive() or monotonic()>=deadline:
                    raise RuntimeError("gateway startup failed")
                sleep(.02)
            yield sock.getsockname()[1]
        finally:
            server.should_exit=True
            thread.join(10)
            (output/"policy_metrics.json").write_bytes(encode({**metrics,"closure_proxy":None if arm=="B6_paac" else float(boundary.totals()[0]),
                                                        "price":flow.get("price",0) if arm=="B9_joint" else None,
                                                        "token_price":flow.get("token_price",0) if arm=="B9_joint" else None}))
