"""Hardware Abstraction Layer: the sim <-> real boundary.

Concrete plants and observers live here. Layers above import only the Protocols
in ``interfaces``; they must never import a concrete plant module directly.
"""
