#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCMS hierarchy for original MPC_planner3's initial trajectory seed.

Publish: /mpc_rcms_control_seq3 (drone_msgs/MPCControlSeq)
Modes: /mpc_rcms_mode3, /mpc_rcms_diagnostics3
"""
from mpc_rcms_runtime import run

if __name__ == '__main__':
    run(3)
