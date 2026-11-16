"""Global CUDA event identities and their statically inferred source streams."""
from dataclasses import dataclass
from .dsl_ast import Command, Group


@dataclass
class Event(Command):
    rank: int
    stream: int
    event: int


@dataclass
class Wait(Command):
    rank: int
    stream: int
    event: int


@dataclass
class StreamWait(Command):
    rank: int
    stream: int
    event: int


def event_layout(program):
    """Infer sparse IDs and unique source streams, including wait-only events.

    Stream waits name destinations and cannot infer ownership alone.
    Repeated records on one source use CUDA's latest-record semantics.
    """
    owners = {}
    dependencies = set()
    for thread in program.threads:
        for node in thread.commands:
            for command in (node.commands if isinstance(node, Group) else [node]):
                if isinstance(command, (Event, Wait, StreamWait)):
                    if isinstance(node, Group):
                        raise ValueError("event commands are not allowed inside a group")
                    for field in ("rank", "stream", "event"):
                        value = getattr(command, field)
                        if type(value) is not int or value < 0:
                            raise ValueError(f"event command: {field} must be a non-negative integer")
                    if command.event > 2147483647:
                        raise ValueError("event ID must be in [0,2147483647]")
                if isinstance(command, (Event, Wait)):
                    owner = (command.rank, command.stream)
                    previous = owners.setdefault(command.event, owner)
                    if previous != owner:
                        raise ValueError(f"event {command.event}: conflicting source streams")
                elif isinstance(command, StreamWait):
                    dependencies.add(command.event)
    missing = dependencies - owners.keys()
    if missing:
        raise ValueError(f"event {min(missing)}: stream_wait requires an event or wait source declaration")
    return [dict(id=event, rank=owners[event][0], stream=owners[event][1])
            for event in sorted(owners)]
