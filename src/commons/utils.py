import string
import random
import os
import pickle
import hashlib
import subprocess
import threading


def random_alphabetic(length: int) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def hashing(data):
    if type(data) is not bytes:
        data = pickle.dumps(data)
    return hashlib.sha256(data).hexdigest()


def protobuf_msg_to_dict(message):
    message_dict = {}

    for descriptor in message.DESCRIPTOR.fields:
        key = descriptor.name
        value = getattr(message, descriptor.name)

        if descriptor.label == descriptor.LABEL_REPEATED:
            message_list = []

            for sub_message in value:
                if descriptor.type == descriptor.TYPE_MESSAGE:
                    message_list.append(protobuf_msg_to_dict(sub_message))
                else:
                    message_list.append(sub_message)

            message_dict[key] = message_list
        else:
            if descriptor.type == descriptor.TYPE_MESSAGE:
                message_dict[key] = protobuf_msg_to_dict(value)
            else:
                message_dict[key] = value

    return message_dict


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def get_size(path):
    """Get the size of a file or folder in bytes"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    return get_directory_size(path)


def get_free_space(path):
    command = "df %s | awk '{print $4}' | tail -n 1" % path
    free_space = subprocess.check_output(command, shell=True)
    return int(free_space) * 1e3


def convert_to_bytes(value):
    units = {
        "Ki": 1024,
        "Mi": 1024 ** 2,
        "Gi": 1024 ** 3,
        "Ti": 1024 ** 4,
        "Pi": 1024 ** 5,
    }
    num, unit = value[:-2], value[-2:]
    return int(num) * units[unit]


def delete_files(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            os.remove(path)


class SafeDict:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __len__(self):
        with self._lock:
            return len(self._data)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def clear(self):
        with self._lock:
            self._data.clear()
