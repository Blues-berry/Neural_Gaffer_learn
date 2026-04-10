import argparse
import json
import struct
from pathlib import Path


GLB_HEADER_STRUCT = struct.Struct("<4sII")
GLB_CHUNK_HEADER_STRUCT = struct.Struct("<II")
JSON_CHUNK_TYPE = 0x4E4F534A


def parse_args():
    parser = argparse.ArgumentParser(description="Strip a named extension from a GLB JSON chunk.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extension", required=True)
    return parser.parse_args()


def strip_extension(value, extension_name: str):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "extensions" and isinstance(item, dict):
                nested = {
                    nested_key: strip_extension(nested_value, extension_name)
                    for nested_key, nested_value in item.items()
                    if nested_key != extension_name
                }
                if nested:
                    cleaned[key] = nested
                continue
            cleaned[key] = strip_extension(item, extension_name)
        return cleaned
    if isinstance(value, list):
        return [strip_extension(item, extension_name) for item in value]
    return value


def pad_chunk(data: bytes, pad_byte: bytes) -> bytes:
    padding = (-len(data)) % 4
    if padding == 0:
        return data
    return data + pad_byte * padding


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    blob = input_path.read_bytes()

    magic, version, total_length = GLB_HEADER_STRUCT.unpack_from(blob, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: {input_path}")
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version}: {input_path}")
    if total_length != len(blob):
        raise ValueError(f"Corrupt GLB length for {input_path}")

    chunks = []
    offset = GLB_HEADER_STRUCT.size
    json_payload = None
    while offset < len(blob):
        chunk_length, chunk_type = GLB_CHUNK_HEADER_STRUCT.unpack_from(blob, offset)
        offset += GLB_CHUNK_HEADER_STRUCT.size
        chunk_data = blob[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK_TYPE:
            json_payload = json.loads(chunk_data.decode("utf-8"))
        else:
            chunks.append((chunk_type, chunk_data))

    if json_payload is None:
        raise ValueError(f"No JSON chunk found in {input_path}")

    json_payload = strip_extension(json_payload, args.extension)
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in json_payload:
            filtered = [item for item in json_payload[key] if item != args.extension]
            if filtered:
                json_payload[key] = filtered
            else:
                json_payload.pop(key, None)

    json_chunk = json.dumps(json_payload, separators=(",", ":")).encode("utf-8")
    json_chunk = pad_chunk(json_chunk, b" ")

    output = bytearray()
    output.extend(GLB_HEADER_STRUCT.pack(b"glTF", 2, 0))
    output.extend(GLB_CHUNK_HEADER_STRUCT.pack(len(json_chunk), JSON_CHUNK_TYPE))
    output.extend(json_chunk)
    for chunk_type, chunk_data in chunks:
        chunk_data = pad_chunk(chunk_data, b"\x00")
        output.extend(GLB_CHUNK_HEADER_STRUCT.pack(len(chunk_data), chunk_type))
        output.extend(chunk_data)

    output[8:12] = struct.pack("<I", len(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
