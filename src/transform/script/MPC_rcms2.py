#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCMS hierarchy for original MPC_planner2's initial trajectory seed.

Publish: /mpc_rcms_control_seq2 (drone_msgs/MPCControlSeq)
Modes: /mpc_rcms_mode2, /mpc_rcms_diagnostics2
"""
from mpc_rcms_runtime import run

if __name__ == '__main__':
    run(2)
