"""Stand-ins for the two basicsr helpers the vendored CodeFormer files import."""
import logging


def get_root_logger():
    return logging.getLogger("codeformer")


class _Registry:
    def register(self, obj=None):
        return obj if obj is not None else (lambda o: o)


ARCH_REGISTRY = _Registry()
