"""Offline structural checks (no rospy required)."""
from pathlib import Path
import ast
root=Path(__file__).parent
for name in ('MPC_parseh0.py','MPC_parseh1.py','MPC_parseh2.py','MPC_parseh3.py',
             'mpc_parseh_runtime.py','MPC_parseh_selector.py','parseh_accel.py'):
    ast.parse((root/name).read_text(),filename=name)
worker=(root/'mpc_parseh_runtime.py').read_text()
selector=(root/'MPC_parseh_selector.py').read_text()
assert 'ParsehHierarchy(' not in worker
assert 'node.planner.solve = hierarchy.solve' not in worker
assert 'all_failed' in selector
assert 'self.backend.evaluate' in selector
print('Structural checks passed')
