from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Generic, TypeVar
from abc import ABC
from parsy import from_enum, regex, seq, string, decimal_digit
from functools import partial

T = TypeVar('T')

class BufferSize(Enum):
    BufferSize = None
    def __str__(self):
        return "size"

class NonEmptyList(Generic[T]):
    def __init__(self, head: T, *tail: T):
        self.items = [head] + list(tail)

class BufferType(Enum):
    Int = "int"
    Float = "float"

@dataclass
class Metadata():
    kind: BufferType
    size: int | BufferSize

@dataclass
class Channel():
    communicator: int
    stream: int

@dataclass(frozen=True)
class Context:
    indent_level: int
    ranks: int
    buffers: int
    current_device: int

class ReduceOp(Enum):
    Sum = "sum"
    Product = "product"

class CollectiveFunction(Enum):
    AllReduce = "allReduce"
    Broadcast = "broadcast"
    Reduce = "reduce"
    AllGather = "allGather"
    ReduceScatter = "reduceScatter"
    AllToAll = "allToAll"
    Gather = "gather"
    Scatter = "scatter"

    @property
    def ncclString(self):
        mapping = {
            CollectiveFunction.AllReduce: "ncclAllReduce",
            CollectiveFunction.Reduce: "ncclReduce",
            CollectiveFunction.AllGather: "ncclAllGather",
            CollectiveFunction.ReduceScatter: "ncclReduceScatter",
            CollectiveFunction.AllToAll: "ncclAllToAll",
            CollectiveFunction.Gather: "ncclGather",
            CollectiveFunction.Scatter: "ncclScatter",
        }
        return mapping[self]

class PointToPointFunction(Enum):
    Send = "send"
    Receive = "recv"

    @property
    def ncclString(self):
        mapping = {
            PointToPointFunction.Send: "ncclSend",
            PointToPointFunction.Receive: "ncclRecv",
        }
        return mapping[self]

    
@dataclass
class Node(ABC):
    pass

@dataclass
class Command(Node):
    pass

@dataclass
class Collective(Command):
    rank: int
    channel: Channel
    source_buffer: int | None
    destination_buffer: int | None
    meta: Metadata

@dataclass
class Rooted:
    root: int

@dataclass
class Reductive:
    op: ReduceOp

@dataclass
class AllReduce(Collective, Reductive):
    pass

@dataclass
class Broadcast(Collective, Rooted):
    pass

@dataclass
class Reduce(Collective, Rooted, Reductive):
    pass

@dataclass 
class AllGather(Collective):
    pass

@dataclass
class ReduceScatter(Collective, Reductive):
    pass

@dataclass
class AllToAll(Collective):
    pass

@dataclass
class Gather(Collective, Rooted):
    pass

@dataclass
class Scatter(Collective, Rooted):
    pass

@dataclass
class PointToPoint(Command):
    rank: int
    peer: int
    channel: Channel
    buffer: int
    meta: Metadata

@dataclass
class Send(PointToPoint):
    pass

@dataclass
class Recv(PointToPoint):
    pass

@dataclass
class Barrier(Command):
    participants: tuple[int, ...]

    def __post_init__(self):
        if not self.participants or any(type(t) is not int or t < 0 for t in self.participants):
            raise ValueError("barrier requires a nonempty set of non-negative thread IDs")
        self.participants = tuple(sorted(set(self.participants)))


@dataclass
class Sync(Command):
    rank: int
    stream: int

@dataclass
class Fill(Command):
    rank: int
    stream: int
    buffer: int
    meta: Metadata
    value: float
    delay_cycles: int

@dataclass
class Copy(Command):
    rank: int
    stream: int
    source_buffer: int
    destination_buffer: int
    meta: Metadata

@dataclass
class Group(Node):
    commands: NonEmptyList(Command)

@dataclass
class Thread(Node):
    commands: NonEmptyList(Command | Group)

@dataclass
class Program(Node):
    threads: NonEmptyList(Thread)


def validate_barriers(program: Program) -> Program:
    """Check local barrier well-formedness, without requiring progress."""
    def visit(node, caller, in_group=False):
        if isinstance(node, Group):
            for command in node.commands:
                visit(command, caller, True)
        elif isinstance(node, Barrier):
            node.__post_init__()
            if in_group:
                raise ValueError("barrier is not allowed inside a group")
            if caller not in node.participants:
                raise ValueError("barrier participant set must contain the calling thread")
            if any(t >= len(program.threads) for t in node.participants):
                raise ValueError("barrier participant does not name an existing thread")

    for caller, thread in enumerate(program.threads):
        for node in thread.commands:
            visit(node, caller)
    return program


@dataclass
class Skip(Command):
    pass


class RootRole:
    """Root endpoint: issuing rank is root and both buffers are meaningful."""


class PeerRole:
    """Peer endpoint: only the directionally meaningful buffer is present."""


class BroadcastRoot(Broadcast, RootRole):
    pass


class BroadcastPeer(Broadcast, PeerRole):
    pass


class ScatterRoot(Scatter, RootRole):
    pass


class ScatterPeer(Scatter, PeerRole):
    pass


class GatherRoot(Gather, RootRole):
    pass


class GatherPeer(Gather, PeerRole):
    pass


class ReduceRoot(Reduce, RootRole):
    pass


class ReducePeer(Reduce, PeerRole):
    pass


def validate_collective_role(command):
    """Enforce endpoint roles even for callers constructing AST nodes directly."""
    if isinstance(command, RootRole):
        if command.rank != command.root:
            raise ValueError("root endpoint must issue on the root")
        active = (command.source_buffer, command.destination_buffer)
    elif isinstance(command, PeerRole):
        if command.rank == command.root:
            raise ValueError("peer endpoint must differ from root")
        receiving = isinstance(command, (Broadcast, Scatter))
        inactive = command.source_buffer if receiving else command.destination_buffer
        if inactive is not None:
            raise ValueError("peer endpoint has an inactive buffer")
        active = (command.destination_buffer if receiving else command.source_buffer,)
    else:
        return
    if any(type(buffer) is not int or buffer < 0 for buffer in active):
        raise ValueError("collective endpoint requires non-negative active buffer indices")
    if any(type(rank) is not int or rank < 0 for rank in (command.rank, command.root)):
        raise ValueError("collective endpoint requires non-negative rank indices")


def validate_collective_roles(program):
    for thread in program.threads:
        for node in thread.commands:
            for command in (node.commands if isinstance(node, Group) else [node]):
                validate_collective_role(command)
    return program
