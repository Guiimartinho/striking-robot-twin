"""Independent safety channel: keep-out, reach, force cap, latency margin.

On the real robot this logic lives on the STM32, separate from perception. Here
it is a layer that can veto or abort any command coming from decision-making.
"""
