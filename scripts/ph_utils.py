import os
import sys
import importlib
import logging
import hashlib
import mmap
from contextlib import contextmanager

from modules import sd_hijack


def maybe_import(module_dir, module_name):
  if module_name in sys.modules:
    existed_module = sys.modules[module_name].__file__
    existed_module = os.path.splitext(existed_module)[0]
    if os.path.exists(
        os.path.join(module_dir, '%s.py' % module_name)) or os.path.exists(
        os.path.join(module_dir, '%s.pyc' % module_name)):
      new_module = os.path.join(module_dir, '%s' % module_name)
    else:
      new_module = os.path.join(module_dir, module_name, '__init__')

    if existed_module == new_module:
      return sys.modules[module_name]
    else:
      raise ValueError('conflict import. Already defined: %s. Try to define: %s' % (existed_module, new_module))

  if module_dir not in sys.path:
    sys.path.append(module_dir)

  try:
    module = importlib.import_module(module_name)
    return module
  except ImportError as e:
    logging.error(e)
    return None


def hash_lora_file(filename):
    """Hashes a .safetensors file using the new hashing method.
    Only hashes the weights of the model."""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")

    with open(filename, mode="rb") as file_obj:
        offset = n + 8
        file_obj.seek(offset)
        for chunk in iter(lambda: file_obj.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()[:12]


@contextmanager
def custom_embeddings_dir():
    custom_embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 'embeddings')
    sd_hijack.model_hijack.embedding_db.add_embedding_dir(custom_embeddings_dir)
    yield
    if custom_embeddings_dir in sd_hijack.model_hijack.embedding_db.embedding_dirs:
       sd_hijack.model_hijack.embedding_db.embedding_dirs.pop(custom_embeddings_dir)
