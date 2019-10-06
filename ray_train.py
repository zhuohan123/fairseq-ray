#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import copy
import socket
import time

import ray
import torch

import fairseq
from fairseq import checkpoint_utils, options
from fairseq_cli.train import main
from contextlib import closing

_original_save_checkpoint = checkpoint_utils.save_checkpoint


class RayDistributedActor:
    def run(self, url, world_rank, args):
        print("Ray worker at {url} rank {rank}".format(url=url, rank=world_rank))
        self.url = url
        self.world_rank = world_rank
        args.distributed_rank = world_rank
        args.distributed_init_method = url

        if args.cpu:
            original_n_cpus = args.distributed_world_size

            def _new_save_checkpoint(*args, **kwargs):
                _original_save_checkpoint(*args, **kwargs)
                n_cpus = int(ray.cluster_resources()["CPU"])
                if n_cpus > original_n_cpus:
                    raise Exception("New CPUs find (original %d CPUs, now %d CPUs)"
                                    % (original_n_cpus, n_cpus))
        else:
            original_n_gpus = args.distributed_world_size

            def _new_save_checkpoint(*args, **kwargs):
                _original_save_checkpoint(*args, **kwargs)
                n_gpus = int(ray.cluster_resources().get("GPU", 0))
                if n_gpus > original_n_gpus:
                    raise Exception("New GPUs find (original %d GPUs, now %d GPUs)"
                                    % (original_n_gpus, n_gpus))
        fairseq.checkpoint_utils.save_checkpoint = _new_save_checkpoint

        main(args, init_distributed=(args.distributed_world_size > 1))

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


def add_ray_args(parser):
    group = parser.add_argument_group('Ray related arguments')
    # fmt: off
    group.add_argument('--ray-address', default="auto", type=str,
                       help='address for ray initialization')
    group.add_argument('--fix-batch-size', default=None, type=int,
                       help='fix batch size (max_sentences * update_freq '
                            '* n_GPUs) to be a fixed input value for different '
                            'number of GPUs or CPUs')
    # fmt: on
    return group


def ray_main():
    parser = options.get_training_parser()
    add_ray_args(parser)
    args = options.parse_args_and_arch(parser)
    original_args = copy.deepcopy(args)
    retry = True
    while retry:
        args = copy.deepcopy(original_args)
        ray.init(address=args.ray_address)
        if args.cpu:
            args.distributed_world_size = int(ray.cluster_resources()["CPU"])
        else:
            n_gpus = int(ray.cluster_resources().get("GPU", 0))
            while n_gpus == 0:
                print("No GPUs available, wait 10 seconds")
                time.sleep(10)
                n_gpus = int(ray.cluster_resources().get("GPU", 0))
            args.distributed_world_size = n_gpus
        if args.fix_batch_size is not None:
            args.update_freq = math.ceil(
                args.fix_batch_size / (args.max_sentences *
                                       args.distributed_world_size))
            print("Training on %d GPUs, max_sentences=%d, update_freq=%d"
                  % (args.distributed_world_size, args.max_sentences,
                     args.fix_batch_size))
        Actor = ray.remote(
            num_cpus=1, num_gpus=int(not args.cpu))(RayDistributedActor)
        workers = [Actor.remote() for i in range(args.distributed_world_size)]
        ip = ray.get(workers[0].get_node_ip.remote())
        port = ray.get(workers[0].find_free_port.remote())
        address = "tcp://{ip}:{port}".format(ip=ip, port=port)
        unfinished = [worker.run.remote(address, i, args)
                      for i, worker in enumerate(workers)]
        try:
            while len(unfinished) > 0:
                finished, unfinished = ray.wait(unfinished)
                finished = ray.get(finished)
            retry = False
        except Exception as inst:
            print("Ray restart because following error occurs:")
            print(inst)
            retry = True
        ray.shutdown()


if __name__ == '__main__':
    ray_main()
