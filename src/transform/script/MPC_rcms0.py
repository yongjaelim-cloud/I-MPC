#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCMS hierarchy for original MPC_planner0's initial trajectory seed.

Publish: /mpc_rcms_control_seq0 (drone_msgs/MPCControlSeq)
Modes: /mpc_rcms_mode0, /mpc_rcms_diagnostics0
"""
from mpc_rcms_runtime import run

if __name__ == '__main__':
    run(0)
