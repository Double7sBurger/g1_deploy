"""Deploy an Isaac Lab G1 locomotion policy on a real Unitree G1, or on MuJoCo over the same DDS.

The robot-facing half needs only numpy, torch and ``unitree_sdk2py``: no simulator, no Isaac Lab, no
GPU. :mod:`g1_deploy.sim` is optional and is the only part that pulls in MuJoCo.
"""

__all__ = ["core", "controller", "hardware", "bootstrap", "benchmark"]
