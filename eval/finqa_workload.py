"""Synthetic financial contracts. Offline equations never enter the routing state."""
import ast
import json
import math
import random
import re
from dataclasses import dataclass

from eval import collab_experiment_runner as R


# Financial definition, semantic columns A/B/C, offline reference, positive domain.
FORMAL = [
 ("gross margin", "revenue,cost of goods sold", "(A-B)/A", "A"),
 ("inventory turnover", "cost of goods sold,opening inventory,closing inventory", "2*A/(B+C)", "B"),
 ("cash conversion cycle", "days sales outstanding,days inventory outstanding,days payable outstanding", "A+B-C", "A"),
 ("current ratio", "current assets,current liabilities", "A/B", "B"),
 ("after-tax operating profit", "operating income,tax rate in percent", "A*(1-B/100)", "A"),
 ("economic value added", "after-tax operating profit,invested capital,cost of capital in percent", "A-B*C/100", "B"),
 ("interest coverage", "operating profit,interest expense", "A/B", "B"),
 ("diluted earnings per share", "net income,preferred dividends,diluted shares", "(A-B)/C", "C"),
 ("compound two-period total return", "period one return in percent,period two return in percent", "(1+A/100)*(1+B/100)-1", "A"),
 ("weighted average unit cost", "opening inventory cost,purchase cost,available units", "(A+B)/C", "C"),
 ("dividend payout ratio", "dividend per share,earnings per share", "A/B", "B"),
 ("free cash flow to equity", "operating cash flow,capital expenditure,net borrowing", "A-B+C", "B"),
 ("net debt to EBITDA", "gross debt,cash,EBITDA", "(A-B)/C", "C"),
 ("annualized days sales outstanding", "receivables,annual credit sales", "365*A/B", "B"),
 ("straight line annual depreciation", "purchase price,salvage value,useful life in years", "(A-B)/C", "C"),
 ("break-even volume", "fixed cost,unit selling price,unit variable cost", "A/(B-C)", "A"),
 ("bond current yield", "annual coupon payment,bond price", "A/B", "B"),
 ("single-period discounted cash flow", "next year cash flow,discount rate in percent", "A/(1+B/100)", "A"),
 ("net present value of a one-year investment", "initial outlay,end cash flow,discount rate in percent", "B/(1+C/100)-A", "A"),
 ("levered equity beta with zero tax", "asset beta,debt,equity", "A*(1+B/C)", "C"),
 ("enterprise value", "equity market value,debt,cash", "A+B-C", "A"),
 ("sustainable growth rate", "return on equity in percent,dividend payout in percent", "A/100*(1-B/100)", "A"),
 ("price to book per share", "stock price,total equity,shares outstanding", "A*C/B", "B"),
 ("two-asset portfolio return", "first weight in percent,first return in percent,second return in percent", "(A*B+(100-A)*C)/10000", "A"),
]
DEV = [
 ("operating expense ratio", "operating expenses,net sales", "A/B", "B"),
 ("net working capital", "current assets,current liabilities", "A-B", "A"),
 ("average asset turnover", "net sales,opening assets,closing assets", "2*A/(B+C)", "B"),
 ("retention ratio", "net income,dividends", "(A-B)/A", "A"),
 ("annual simple interest", "principal,annual rate in percent", "A*B/100", "A"),
 ("net profit margin", "net earnings,net sales", "A/B", "B"),
]

SYMBOLS = "ABC"


def substitute(expression, values):
    if not isinstance(expression, str):
        raise ValueError("symbolic expression required")
    return R.arithmetic(re.sub(r"\b[A-C]\b", lambda m: str(values[m[0]]), expression))


def generate(development):
    rng = random.Random(7401 if development else 7402)
    packets, truth = {}, {}
    for family,(name, columns, formula, guard) in enumerate(DEV if development else FORMAL):
        labels = columns.split(",")
        symbols = SYMBOLS[:len(labels)]
        # Shared, public contract examples exercise bindings without disclosing a live row.
        examples = []
        for j in range(3):
            values = dict(zip(symbols,rng.sample(range(3,80),len(labels))))
            examples.append(dict(inputs=values, result=substitute(formula,values)))
        for index in range(5):
            family_id = ("dev-flow-" if development else "confirm-flow-") + str(family)
            tid = f"{family_id}-{index}"
            values = dict(zip(symbols,rng.sample(range(4,90),len(labels))))
            if index == 4:
                values[guard] = 0
            schema, record = {}, {}
            for symbol,label in zip(symbols,labels):
                key = label.replace(" ","_") if index != 3 else "revised_" + label.replace(" ","_")
                factor = 1 if index != 3 else .01
                schema[symbol] = dict(label=label, field=key, scale=factor)
                record[key] = round(values[symbol]/factor)
            path = f"file-{rng.getrandbits(64):016x}"
            owner = f"entity-{rng.getrandbits(64):016x}"
            question = (f"Calculate {name}. Report ratios/returns as a fraction and other quantities in base units. "
                        f"Return undefined when {guard} is not strictly positive. Use the contract examples for conventions.")
            packets[tid] = dict(question=question, metric=name, schema=schema, guard=guard,
                                contract_examples=examples, owner=owner, files={path:record})
            truth[tid] = dict(family_id=family_id, answer=substitute(formula,values) if index!=4 else "undefined",
                              case=("cold","binding","binding","drift","negative")[index], sensitive_units=[owner,path])
    return packets,truth


PROMPT = (R.PROMPT + " For financial contract tasks use symbolic role names A, B, C in expression. "
          "The local calculator binds the live values. Use unit number for the requested base-unit or ratio output.")


def bindings(task):
    record, = task["files"].values()
    return {k:record[v["field"]] * v["scale"] for k,v in task["schema"].items()}


def expression(raw):
    stripped = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj,dict) and set(obj) <= {"expression","unit","answer"}:
            return obj.get("expression")
    except ValueError:
        pass
    return None


def resolve(raw, task):
    expr = expression(raw)
    if expr and isinstance(expr,str) and re.search(r"\b[A-C]\b",expr):
        try:
            vals = bindings(task)
            # Contract validation runs locally for all arms.
            if vals[task["guard"]] <= 0:
                return "undefined"
            obj = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            obj["expression"] = str(substitute(expr,vals))
            return R.resolve_output(json.dumps(obj))
        except (KeyError, ValueError, SyntaxError, ArithmeticError):
            return ""
    return R.resolve_output(raw)


@dataclass(frozen=True)
class ArithmeticDAG:
    nodes: tuple

    @classmethod
    def compile(cls, expr, symbols):
        if not isinstance(expr,str) or len(expr)>512:
            raise ValueError("invalid skill")
        tree=ast.parse(expr,mode="eval")
        if sum(1 for _ in ast.walk(tree))>128:
            raise ValueError("too many nodes")
        nodes=[]
        def visit(node):
            if isinstance(node,ast.Name) and node.id in symbols:
                item=("role",node.id)
            elif isinstance(node,ast.Constant) and type(node.value) in (int,float):
                item=("const",float(node.value))
            elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                item=(type(node.op).__name__,visit(node.operand))
            elif isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Div,ast.Mult):
                item=(type(node.op).__name__,visit(node.left),visit(node.right))
            else:
                raise ValueError("unsupported DAG node")
            if item not in nodes:
                nodes.append(item)
            return nodes.index(item)
        last=visit(tree.body)
        return cls((tuple(nodes),last))

    def execute(self,values):
        import operator
        ops={"UAdd":operator.pos,"USub":operator.neg,"Add":operator.add,"Sub":operator.sub,
             "Div":operator.truediv,"Mult":operator.mul}
        outputs=[]
        nodes,last=self.nodes
        for op,*args in nodes:
            if op=="role":
                val=values[args[0]]
            elif op=="const":
                val=args[0]
            else:
                val=ops[op](*(outputs[i] for i in args))
            if not math.isfinite(val):
                raise ValueError("nonfinite skill result")
            outputs.append(val)
        return outputs[last]


@dataclass(frozen=True)
class Skill:
    dag: ArithmeticDAG
    schema: dict
    guard: str


class Library:
    def __init__(self):
        self.skills = {}
        self.encountered = set()

    def learn(self, task, raw):
        expr = expression(raw)
        if not isinstance(expr,str) or not re.search(r"\b[A-C]\b",expr):
            return False
        try:
            dag = ArithmeticDAG.compile(expr,task["schema"])
            if not all(math.isclose(dag.execute(x["inputs"]),x["result"],abs_tol=1e-8)
                       for x in task["contract_examples"]):
                return False
            if resolve(raw,task) == "" or bindings(task)[task["guard"]] <= 0:
                return False
        except (ValueError,SyntaxError,ArithmeticError,KeyError):
            return False
        self.skills[task["metric"]] = Skill(dag,task["schema"],task["guard"])
        return True

    def replay(self, task, replan):
        skill = self.skills.get(task["metric"])
        if not skill:
            return None
        changed = skill.schema != task["schema"]
        if changed:
            labels = lambda schema: {k:v["label"] for k,v in schema.items()}
            if replan or labels(task["schema"]) != labels(skill.schema):
                return None
        if skill.guard != task["guard"]:
            return None
        vals = bindings(task)
        if vals[skill.guard] <= 0:
            return "undefined","precondition_block"
        try:
            val = skill.dag.execute(vals)
        except (ValueError,KeyError,ArithmeticError,SyntaxError):
            return None
        return str(val), "schema_patch" if changed else "skill"


class ContractViews(R.Views):
    """Propose an opaque calculation plan. The second message reveals the metric schema."""
    def __init__(self,task,cloud,seed):
        self.task,self.cloud,self.reviewed = task,cloud,{}
        self.public=False
        self.stream=True
        self.rows=[0,1]
        self.aliases=[0,1]
        self.target=int(task["guard"]=="A")
        self.literals={task["owner"],*task["files"]}
        self.weights={"shape":1,"semantics":1,"guard_class":1,"owner":4,"path":3}
        self.safe=dict(question=task["question"],roles={k:v["label"] for k,v in task["schema"].items()},
                       contract_examples=task["contract_examples"])

    def offer(self, action, name, obj, prompt=PROMPT):
        c=self.cloud.candidate(action,name,obj,prompt)
        if name!="full" and any(secret in c.payload.decode() for secret in self.literals):
            return None
        facts={"full":set(self.weights),"relation":{"guard_class"},"typed":{"shape"},
               "report":{"semantics"},"fallback":{"semantics"},"advice":{"shape"}}[name]
        self.reviewed[c.payload,action,R.CAP]=facts
        return c

    def selector(self,structured):
        return self.offer(R.Action.DELEGATE,"relation" if structured else "typed",
                          {"slots":[0,1],"selected":self.target} if structured else {"slots":[0,1],"role_count":len(self.task["schema"])},
                          R.SELECT)

    def report(self,slot,*,fallback=False,history=()):
        return self.offer(R.Action.FALLBACK if fallback else R.Action.DELEGATE,
                          "fallback" if fallback else "report",self.safe)

    def advice(self,structured=False):
        return self.offer(R.Action.ADVISE,"advice",{"roles":list(self.task["schema"])},
                          "Suggest how to verify a financial arithmetic plan locally.")
