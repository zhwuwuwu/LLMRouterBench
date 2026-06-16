import threading

from .generator import DirectGenerator
from .factory import create_generator

# Module-level stop event. Set this (stop_event.set()) from the CLI or runner
# to make all active generator retry loops abort on their next sleep boundary.
stop_event = threading.Event()

__all__ = ['DirectGenerator', 'create_generator', 'stop_event']
