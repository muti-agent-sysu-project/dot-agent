"""Session-scoped lexical views. No grader labels or workspace file access."""
import csv
import io
import json
import re
from uuid import uuid4

FIELDS = {
    "name|person|employee|customer|full_name|patient|姓名|联系人": "PERSON",
    "organization|org|company|employer|公司": "ORG",
    "salary|amount|balance|price|revenue|cost|薪资|金额": "MONEY",
    "account|account_number|iban|card_number|账号": "ACCOUNT",
    "api_key|apikey|access_token|token|password|secret": "SECRET",
    "ssn|id_number|身份证": "ID",
    "email|邮箱": "EMAIL", "phone|tel|电话": "PHONE",
    "date|birth_date|日期": "DATE", "path|filename|file_path": "PATH",
}
FIELD = {k: kind for keys, kind in FIELDS.items() for k in keys.split("|")}
RULES = [
    ("URL", r"https?://[^\s<>\"']+"),
    ("SECRET", r"(?i:\b(?:bearer\s+|(?:api[_ -]?key|password|secret|token)\s*[:=]\s*[\"']?)[A-Za-z0-9._~+/=-]+|\b(?:sk[-]|ghp_|AKIA)[A-Za-z0-9_-]{6,})"),
    ("PATH", r"(?:[A-Za-z]:[\\/]|(?<!\w)/(?:[\w.-]+/)+)[^\s\"'<>|,;]*"),
    ("FILE", r"(?<![\w.])(?:[\w.-]+/)*[\w.-]+\.(?i:csv|xlsx?|docx?|pdf|jsonl?|tsv|txt|log|ya?ml|pptx?|py|md)(?!\w)"),
    ("EMAIL", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    ("ID", r"(?<!\d)(?:\d{17}[\dXx]|\d{3}-\d{2}-\d{4})(?!\d)"),
    ("DATE", r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/\d{4})\b"),
    ("ACCOUNT", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    ("PHONE", r"(?<!\w)(?:\+?\d{1,3}[- .]?)?(?:\(\d{3}\)|\d{3})[- .]?\d{3}[- .]?\d{4}(?!\w)|(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("MONEY", r"(?:[$¥￥€£]\s*|(?:USD|CNY|RMB)\s+)\d[\d,.]*"),
    ("ORG", r"\b(?:[A-Z][\w&-]*\s+){1,4}(?:Inc|LLC|Ltd|Corp)\b|[\u4e00-\u9fff]{2,20}(?:有限公司|集团)"),
    ("PERSON", r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,2}\b"),
    ("NUMBER", r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)"),
]
PATTERN = re.compile("|".join(f"(?P<{name}>{expr})" for name, expr in RULES))
ALIAS = re.compile(r"__EC2_[a-f0-9]{12}_[A-Z]+_\d+__")


def protocol_name(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", value)) and not PATTERN.search(value)


def field_kind(key):
    normalized = re.sub(r"[\s-]+","_",re.sub(r"([a-z])([A-Z])",r"\1_\2",key.strip()).lower())
    if normalized in FIELD:
        return FIELD[normalized]
    for suffix in ("_name","_email","_phone","_salary","_date","_token","_path","_amount"):
        if normalized.endswith(suffix):
            if suffix == "_name" and normalized.startswith(("org","company","vendor","employer")):
                return "ORG"
            return FIELD[suffix[1:]]
    return None


class SessionView:
    def __init__(self):
        self.prefix = uuid4().hex[:12]
        self.forward = {}
        self.reverse = {}
        self.words = {}
        self.characters = 0
        self.protected = {}

    def alias(self, value, kind):
        key = (type(value), value)
        if key not in self.forward:
            width = len(value) if isinstance(value,str) else 0
            if width>4096 or len(self.forward)>=32768 or self.characters+width>1_048_576:
                raise ValueError("session registry limit")
            self.characters += width
            token = f"__EC2_{self.prefix}_{kind}_{len(self.forward)}__"
            self.forward[key] = token
            self.reverse[token] = value
            if kind not in {"CALL","NUMBER"} and len(str(value))>=5:
                node=self.protected
                for char in str(value):
                    node=node.setdefault(char,{})
                node[None]=True
            if isinstance(value, str) and value and not PATTERN.fullmatch(value):
                branch = self.words
                for char in value:
                    branch = branch.setdefault(char, {})
                branch[None] = True
        return self.forward[key]

    def check_envelope(self,obj):
        """Scan the final decoded envelope, including schema keys and metadata."""
        if isinstance(obj,dict):
            for key,value in obj.items():
                self.check_envelope(key)
                self.check_envelope(value)
        elif isinstance(obj,list):
            for value in obj:
                self.check_envelope(value)
        elif isinstance(obj,str):
            text=ALIAS.sub("",obj)
            for start,char in enumerate(text):
                if char not in self.protected:
                    continue
                node=self.protected
                for end in range(start,len(text)):
                    node=node.get(text[end])
                    if node is None:
                        break
                    if None in node:
                        raise ValueError("registered literal in outbound envelope")

    def mask(self, obj, key=""):
        if isinstance(obj,str) and "__EC2_" in obj:
            raise ValueError("reserved marker in local input")
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Keys can themselves be customer names or email addresses.
                result[self.mask(k)] = self.mask(v, k)
            return result
        if isinstance(obj, list):
            return [self.mask(v, key) for v in obj]
        if (kind := field_kind(key)) and type(obj) in (int, float, str) and obj != "":
            return self.alias(obj, kind)
        if type(obj) in (int,float):
            return self.alias(obj,"NUMBER")
        if not isinstance(obj, str):
            return obj
        quoted_path = re.search(r"""(["'])((?:[A-Za-z]:[\\/]|/)[^"'\r\n]+)\1""",obj)
        if quoted_path:
            return (self.mask(obj[:quoted_path.start(2)]) + self.alias(quoted_path[2],"PATH") +
                    self.mask(obj[quoted_path.end(2):]))
        secret = re.search(r"(?i)(?:\bbearer\s+|\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*[\"']?)([A-Za-z0-9._~+/=-]+)", obj)
        if secret:
            return (self.mask(obj[:secret.start(1)]) + self.alias(secret[1],"SECRET") +
                    self.mask(obj[secret.end(1):]))
        # Labelled text in CSV-like notes and plain tool output.
        labelled = re.compile(r"(?i)(?P<label>姓名|联系人|公司|薪资|账号|employee|person|company|salary|account|name)\s*[:=：]\s*(?P<value>[^\n,;，；\t]{1,90})")
        if match := labelled.search(obj):
            val = match["value"].strip()
            if val:
                return (self.mask(obj[:match.start("value")]) +
                        self.alias(val, FIELD[match["label"].casefold()]) + self.mask(obj[match.end():]))
        if (str,obj) in self.forward:
            return self.forward[str,obj]
        # Earliest/longest span wins, so a known name inside a new email never
        # splits the address and leaves a raw domain on the wire.
        spans = [(m.start(), m.end(), m.lastgroup) for m in PATTERN.finditer(obj)]
        for start, char in enumerate(obj):
            if char not in self.words or start and (obj[start-1].isalnum() or obj[start-1]=="_"):
                continue
            branch = self.words
            for end in range(start,len(obj)):
                branch = branch.get(obj[end])
                if branch is None:
                    break
                after = obj[end+1:end+2]
                if None in branch and not (after.isalnum() or after=="_"):
                    spans.append((start,end+1,None))
        parts, at = [], 0
        for start, end, kind in sorted(spans, key=lambda hit: (hit[0], -hit[1])):
            if start < at:
                continue
            raw = obj[start:end]
            parts.extend((obj[at:start], self.alias(raw, kind) if kind else self.forward[(str, raw)]))
            at = end
        return "".join(parts) + obj[at:]

    def restore(self, obj, *, command=False):
        if isinstance(obj, list):
            return [self.restore(v, command=command) for v in obj]
        if isinstance(obj, dict):
            return {self.restore(k): self.restore(v, command=command or k in {"cmd","command","code","script"})
                    for k, v in obj.items()}
        if not isinstance(obj, str):
            return obj
        if "__EC2_" in ALIAS.sub("",obj):
            raise ValueError("malformed alias")
        def value(token):
            if token not in self.reverse:
                raise ValueError("unknown alias")
            raw = self.reverse[token]
            if command and (str(raw).startswith("-") or not re.fullmatch(r"[\w./:@+-]+", str(raw))):
                raise ValueError("unsafe command substitution")
            return raw
        if ALIAS.fullmatch(obj):
            return value(obj)
        return ALIAS.sub(lambda m: str(value(m[0])), obj)

    def table(self,text):
        """Preserve a rectangular CSV/TSV and bind cells by recognized column names."""
        first,sep,rest = text.partition("\n")
        if not sep or not rest.strip():
            return None
        for delimiter in (",","\t"):
            if delimiter not in first:
                continue
            try:
                rows=list(csv.reader(io.StringIO(text),delimiter=delimiter,strict=True))
            except csv.Error:
                continue
            header=rows[0]
            if not any(field_kind(k) for k in header) or any(len(row)!=len(header) for row in rows[1:]):
                continue
            out=io.StringIO()
            writer=csv.writer(out,delimiter=delimiter,lineterminator="\n")
            writer.writerow([self.mask(k) for k in header])
            writer.writerows([self.mask(cell,k) for k,cell in zip(header,row)] for row in rows[1:])
            return out.getvalue()
        return None

    def messages(self, request):
        result = []
        for source in request["messages"]:
            msg = {}
            for key, v in source.items():
                if key == "tool_calls":
                    msg[key] = []
                    for call in v:
                        fn = call["function"]
                        if not protocol_name(fn["name"]):
                            raise ValueError("invalid function name")
                        args = self.mask(json.loads(fn["arguments"]))
                        msg[key].append({"id": self.alias(call["id"], "CALL"), "type": "function",
                                         "function": {"name": fn["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
                elif key == "tool_call_id":
                    msg[key] = self.alias(v, "CALL")
                elif key == "content":
                    if isinstance(v, list):
                        if any(p.get("type") != "text" for p in v):
                            raise ValueError("requires local multimodal processing")
                        msg[key] = [{"type":"text","text":self.mask(p["text"])} for p in v]
                    elif isinstance(v, str) and v.lstrip().startswith(("{","[")):
                        try:
                            parsed = json.loads(v)
                        except ValueError:
                            msg[key] = self.mask(v)
                        else:
                            msg[key] = json.dumps(self.mask(parsed), ensure_ascii=False)
                    else:
                        table = self.table(v) if isinstance(v,str) and source.get("role")=="tool" else None
                        msg[key] = self.mask(v) if table is None else table
                elif key == "role":
                    if v not in {"user","assistant","tool","system","developer"}:
                        raise ValueError("invalid role")
                    msg[key] = v
                elif key == "name":
                    msg[key] = v if source.get("role") == "tool" and protocol_name(v) else self.alias(v,"PERSON")
            result.append(msg)
        return result

    def schema_key(self,key):
        if re.fullmatch(r"[\w-]{1,100}",key) and not PATTERN.search(key):
            return key
        return self.alias(key,"FIELD")

    def tools(self, tools):
        # Function names, parameter names and JSON Schema types are protocol, not entity labels.
        def schema(node):
            if isinstance(node, list):
                return [schema(v) for v in node]
            if isinstance(node, dict):
                result = {}
                for k, v in node.items():
                    if k in {"description","title","default","enum","examples","const"}:
                        result[k] = self.mask(v)
                    elif k == "$schema" and v in {"http://json-schema.org/draft-07/schema#",
                            "https://json-schema.org/draft/2020-12/schema"}:
                        result[k]=v
                    elif k == "$ref":
                        if not isinstance(v,str) or not re.fullmatch(r"#/(?:[$\w-]+/)*[$\w-]+",v):
                            raise ValueError("nonlocal schema reference")
                        result[k] = v
                    elif k == "patternProperties" and set(v) <= {".*","^.*$","^(.*)$"}:
                        result[k] = {pattern:schema(spec) for pattern,spec in v.items()}
                    elif k == "properties":
                        result[k] = {self.schema_key(name):schema(spec) for name,spec in v.items()}
                    elif k == "required" and isinstance(v,list):
                        result[k] = [self.schema_key(name) for name in v]
                    elif re.fullmatch(r"[$\w-]{1,100}", k) and not PATTERN.search(k):
                        result[k] = schema(v)
                    else:
                        match=PATTERN.search(k)
                        reason=match.lastgroup if match else "syntax"
                        raise ValueError(f"unsupported schema key ({reason}, length {len(k)})")
                return result
            if isinstance(node,str) and (not re.fullmatch(r"[\w-]{1,100}",node) or PATTERN.search(node)):
                raise ValueError("unsupported schema literal")
            return node
        for tool in tools:
            if tool.get("type") != "function" or not protocol_name(tool["function"]["name"]):
                raise ValueError("unsupported tool signature")
        return schema(tools)
