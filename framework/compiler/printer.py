from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC 
from parsy import from_enum, regex, seq, string, decimal_digit
from functools import partial
from .dsl_ast import *
from .events import Event, Wait, StreamWait, event_layout

def ncclString(command: Command):
    match command:
        case AllReduce():
            return "ncclAllReduce"
        case Broadcast():
            return "ncclBroadcast"
        case Gather():
            return "ncclGather"
        case Scatter():
            return "ncclScatter"
        case AllGather():
            return "ncclAllGather"
        case ReduceScatter():
            return "ncclReduceScatter"
        case Reduce():
            return "ncclReduce"
        case AllToAll():
            return "ncclAlltoAll"
        case Send():
            return "ncclSend"
        case Recv():
            return "ncclRecv"

class Printer:
    def __init__(self, streams_per_gpu: int, communicators=None):
        self.current_device = -1
        self.communicators = communicators
        self.streams_per_gpu = streams_per_gpu

    def _stream_index(self, device: int, stream: int) -> str:
        return f"getStream({device}, {stream})"

    def _select_device(self, device: int) -> str:
        if device == self.current_device:
            return ""
        self.current_device = device
        return f"CUDACHECK(cudaSetDevice({device}));\n"

    def print_program(self, program: Program) -> str:
        validate_barriers(program)
        validate_collective_roles(program)
        event_layout(program)
        participants = sorted({node.participants for thread in program.threads
                               for node in thread.commands if isinstance(node, Barrier)})
        self.barriers = {ts: f"host_barrier_{i}" for i, ts in enumerate(participants)}
        num_threads = len(program.threads)
        
        output = [
            f"std::barrier sync_point({num_threads});",
            *(f"std::barrier {name}({len(ts)});" for ts, name in self.barriers.items()),
            "std::vector<std::jthread> threads;"
        ]
        
        try:
            for thread in program.threads:
                output.append(self.print_thread(thread))
            return "\n".join(output)
        finally:
            # Standalone visitors must not reuse a previous program's objects.
            self.barriers = {}

    def print_thread(self, thread: Thread) -> str:
        self.current_device = -1 
        
        output = [
            "threads.emplace_back([&]() {",
            "    sync_point.arrive_and_wait();"
        ]
        
        for node in thread.commands:
            cmd_out = self.print_command(node)
            
            indented = "\n".join(f"    {line}" for line in cmd_out.split("\n"))
            output.append(indented)
            
        output.append("});")
        return "\n".join(output)

    def print_group(self, group: Group) -> str:
        if any(not isinstance(command, (Collective, PointToPoint, Skip)) for command in group.commands):
            raise ValueError("only NCCL commands and skip are allowed inside a group; barrier is not allowed inside a group")
        output = []
        output.append("NCCLCHECK(ncclGroupStart());")
        for command in group.commands:
            output.append(self.print_command(command))
        output.append("NCCLCHECK(ncclGroupEnd());")
        return "\n".join(output)

    def print_command(self, command: Command) -> str:
        match command:
            case PointToPoint() as command:
                return self.print_send_recv(command)
            case Collective() as command:
                return self.print_collective(command)
            case Barrier() as command:
                if command.participants not in getattr(self, "barriers", {}):
                    raise ValueError("barrier printing requires a validated program context")
                return f"{self.barriers[command.participants]}.arrive_and_wait();"
            case Skip():
                return ";"
            case Sync() as command:
                return self.print_sync(command)
            case Event() | Wait() | StreamWait() as command:
                return self.print_event(command)
            case Fill() as command:
                return self.print_fill(command)
            case Copy() as command:
                return self.print_copy(command)
            case Group() as group:
                return self.print_group(group)
            case _:
                raise TypeError(f"Unknown command type: {type(command)}")

    def print_send_recv(self, command: PointToPoint) -> str:
        return self._generate_output(command, is_point_to_point=True)

    def print_sync(self, command: Sync) -> str:
        return (
            self._select_device(command.rank)
            + f"CUDACHECK(cudaStreamSynchronize({self._stream_index(command.rank, command.stream)}));"
        )

    def print_collective(
        self,
        command: Collective,
    ) -> str:
        validate_collective_role(command)
        return self._generate_output(
            command,
            is_point_to_point=False,
            is_rooted=isinstance(command, Rooted),
            is_reductive=isinstance(command, Reductive)
        )

    def print_event(self, command):
        stream = self._stream_index(command.rank, command.stream)
        event = f"getEvent({command.event})"
        if isinstance(command, Event):
            call = f"cudaEventRecord({event}, {stream})"
        elif isinstance(command, Wait):
            call = f"cudaEventSynchronize({event})"
        else:
            call = f"cudaStreamWaitEvent({stream}, {event}, 0)"
        return self._select_device(command.rank) + f"CUDACHECK({call});"

    def _comm_rank(self, communicator, rank):
        # Standalone printing without a layout assumes identity rank numbering.
        if self.communicators is None:
            return rank
        return self.communicators[communicator].index(rank)

    def print_fill(self, command):
        if command.meta.kind != BufferType.Float:
            raise ValueError("fill supports float storage only")
        count = str(command.meta.size)
        literal = repr(command.value) + "f"
        return (self._select_device(command.rank)
                + f"launchFillBuffer(devBuffers[{command.rank}][{command.buffer}], "
                  f"{count}, {literal}, {command.delay_cycles}ULL, "
                  f"{self._stream_index(command.rank, command.stream)});\n"
                  "CUDACHECK(cudaGetLastError());")

    def print_copy(self, command):
        if command.meta.kind != BufferType.Float:
            raise ValueError("copy supports float storage only")
        return (self._select_device(command.rank)
                + f"CUDACHECK(cudaMemcpyAsync(devBuffers[{command.rank}][{command.destination_buffer}], "
                  f"devBuffers[{command.rank}][{command.source_buffer}], "
                  f"{command.meta.size} * sizeof(float), cudaMemcpyDeviceToDevice, "
                  f"{self._stream_index(command.rank, command.stream)}));")

    def _generate_output(
        self,
        command: Command,
        is_point_to_point: bool = False,
        is_rooted: bool = False,
        is_reductive: bool = False
    ) -> str:
        
        device = command.rank
        cuda_set_device_str = self._select_device(device)
        
        buffer_type_str = command.meta.kind.value
        buffer_size = command.meta.size
        rank = command.rank
        
        nccl_comm = f"getComm({command.rank}, {command.channel.communicator})"
        nccl_stream = self._stream_index(device, command.channel.stream)

        
        if is_point_to_point:
            peer_rank = self._comm_rank(command.channel.communicator, command.peer)
            buffer_ptr = f"devBuffers[{device}][{command.buffer}]"
            
            return (
                f"{cuda_set_device_str}"
                f"NCCLCHECK({ncclString(command)}({buffer_ptr}, {buffer_size}, nccl{buffer_type_str.capitalize()}, {peer_rank}, {nccl_comm}, {nccl_stream}));"
            )
        else:
            # Unused peer arguments alias the active buffer to satisfy NCCL
            # pointer checking modes without introducing a second DSL buffer.
            source = (command.source_buffer if command.source_buffer is not None
                      else command.destination_buffer)
            destination = (command.destination_buffer if command.destination_buffer is not None
                           else command.source_buffer)
            source_ptr = f"devBuffers[{device}][{source}]"
            destination_ptr = f"devBuffers[{device}][{destination}]"
            
            args = [
                source_ptr,
                destination_ptr,
                str(buffer_size),
                f"nccl{buffer_type_str.capitalize()}",
            ]

            # NCCL argument order is (..., datatype, [op], [root], comm, stream),
            # e.g. ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream).
            if is_reductive:
                op_map = {
                    ReduceOp.Sum: "ncclSum",
                    ReduceOp.Product: "ncclProd"
                }
                op = op_map.get(command.op)
                if op is None:
                    raise ValueError(f"Unknown reduce operation: {command.op}")
                args.append(op)

            if is_rooted:
                root = self._comm_rank(command.channel.communicator, command.root)
                args.append(str(root))

            args.extend([nccl_comm, nccl_stream])
            
            return (
                f"{cuda_set_device_str}"
                f"NCCLCHECK({ncclString(command)}({', '.join(args)}));"
            )
