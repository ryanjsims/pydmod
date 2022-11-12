from io import BytesIO
import logging

logger = logging.getLogger("utils")

def read_cstr(data: BytesIO) -> str:
    value = data.read(1)
    while value[-1] != 0:
        value += data.read(1)
    try:
        string = str(value.strip(b'\0'), encoding='utf-8')
    except UnicodeDecodeError as e:
        display = value.strip(b'\0')
        logger.error(f"Failed to decode bytes {display}")
        raise e
    return string