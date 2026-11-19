"""Minimal inference shim for a training-only DeepSpeed registration call."""
class _Zero:
    @staticmethod
    def register_external_parameter(_module, _parameter):
        return None


zero = _Zero()

