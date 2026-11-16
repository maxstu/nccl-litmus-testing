"""Paper-shaped syntax; execution metadata follows the paper's context arguments."""
from parsy import from_enum, regex, seq, string
from .dsl_ast import *
from .events import Event, Wait, StreamWait, event_layout

standard_whitespace = regex(r"\s+")
line_comment = regex(r"//[^\n]*")
inline_comment = regex(r"/\*[\s\S]*?\*/")
whitespace = (standard_whitespace | line_comment | inline_comment).many()
lexeme = lambda p: p << whitespace
num = lexeme(regex(r"\d+")).map(int)
value = lexeme(regex(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")).map(float)
size = num | lexeme(string("buffer_size")).map(lambda _: BufferSize.BufferSize)
comma, semicolon = lexeme(string(",")), lexeme(string(";"))
in_paren = lambda p: lexeme(string("(")) >> p << lexeme(string(")"))
in_square = lambda p: lexeme(string("[")) >> p << lexeme(string("]"))
in_angle = lambda p: lexeme(string("<")) >> p << lexeme(string(">"))
in_brace = lambda p: lexeme(string("{")) >> p << lexeme(string("}"))
kind = lexeme(from_enum(BufferType))
reduce_op = lexeme(from_enum(ReduceOp))
meta = seq(kind=kind << comma, size=size).combine_dict(Metadata)
channel = seq(communicator=num << comma, stream=num).combine_dict(Channel)
buffers = in_square(seq(source_buffer=num << comma, destination_buffer=num))


def collective_parser(name, cls, reductive=False, role=None, receiving=False):
    ranks = in_angle(seq(rank=num << comma, root=num) if role == "peer" else seq(rank=num))
    bufs = (in_square(seq(buffer=num)) if role == "peer" else buffers)
    args = {"channel": channel << comma}
    if reductive:
        args["op"] = reduce_op << comma
    args["meta"] = meta
    context = in_paren(seq(**args))

    def build(parts):
        fields = {**parts[0], **parts[1], **parts[2]}
        if role == "root":
            fields["root"] = fields["rank"]
        if role == "peer":
            buf = fields.pop("buffer")
            fields["source_buffer"] = None if receiving else buf
            fields["destination_buffer"] = buf if receiving else None
            if fields["rank"] == fields["root"]:
                raise ValueError("peer endpoint must differ from root; use the Root command")
        return cls(**fields)

    return lexeme(string(name)) >> seq(ranks, bufs, context).map(build)


collective = (
    collective_parser("allReduce", AllReduce, True)
    | collective_parser("reduceScatter", ReduceScatter, True)
    | collective_parser("allGather", AllGather)
    | collective_parser("allToAll", AllToAll)
)
for name, root_cls, peer_cls, reductive, receiving in (
    ("broadcast", BroadcastRoot, BroadcastPeer, False, True),
    ("scatter", ScatterRoot, ScatterPeer, False, True),
    ("gather", GatherRoot, GatherPeer, False, False),
    ("reduce", ReduceRoot, ReducePeer, True, False),
):
    collective |= collective_parser(name + "Root", root_cls, reductive, "root")
    collective |= collective_parser(name + "Peer", peer_cls, reductive, "peer", receiving)


def p2p_parser(name, cls):
    return lexeme(string(name)) >> seq(
        in_angle(seq(rank=num << comma, peer=num)),
        in_square(seq(buffer=num)),
        in_paren(seq(channel=channel << comma, meta=meta)),
    ).map(lambda parts: cls(**{key: value for part in parts for key, value in part.items()}))


nccl_command = collective | p2p_parser("send", Send) | p2p_parser("recv", Recv)
sync = lexeme(string("sync")) >> in_paren(seq(rank=num << comma, stream=num)).combine_dict(Sync)
event_args = in_paren(seq(rank=num << comma, stream=num << comma, event=num))
event = lexeme(string("event")) >> event_args.combine_dict(Event)
wait = lexeme(string("wait")) >> event_args.combine_dict(Wait)
stream_wait = lexeme(string("stream_wait")) >> event_args.combine_dict(StreamWait)
rank_stream = in_angle(seq(rank=num << comma, stream=num))
fill = lexeme(string("fill")) >> seq(
    rank_stream, in_square(seq(buffer=num)),
    in_paren(seq(meta=meta << comma, value=value << comma, delay_cycles=num)),
).map(lambda parts: Fill(**{key: value for part in parts for key, value in part.items()}))
copy = lexeme(string("copy")) >> seq(rank_stream, buffers, in_paren(seq(meta=meta))).map(
    lambda parts: Copy(**{key: value for part in parts for key, value in part.items()}))
barrier = lexeme(string("barrier")) >> in_paren(in_brace(num.sep_by(comma))).map(Barrier)
skip = lexeme(string("skip")).result(Skip())
command = nccl_command | barrier | sync | event | wait | stream_wait | fill | copy | skip
sequential = lambda p: p.sep_by(semicolon) << semicolon.optional()
group = lexeme(string("group")) >> in_brace(sequential(nccl_command | skip).map(Group))
body = sequential(command | group)
thread = lexeme(string("thread")) >> in_brace(body.map(Thread))


def validate_program(ast):
    validate_barriers(ast)
    validate_collective_roles(ast)
    event_layout(ast)
    return ast


program = whitespace >> (thread.at_least(1) | body.map(Thread).map(lambda t: [t])).map(Program).map(validate_program)
