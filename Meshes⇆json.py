#!/usr/bin/env python3
# ============================================================================
# UNIVERSAL EDITION + JSON TRANSLATION (container versions 0x33 .. 0x3D)
# ============================================================================
# Decodes and re-encodes every known layout:
#   0x33        empty levels, legacy LOD0
#   0x34-0x36   legacy LOD0: generic meshes + terrain + cloud + skirts/occluder
#   0x36/0x37   adds the METR section; 0x37 also has a cloud proxy mesh
#   0x38-0x3D   decoded GEO0 terrain; LOD0 carries no terrain
#   0x3C-0x3D   mask-less cloud with four packed blobs
#
# JSON layout (format "sky-meshes-v2"):
#   version / version_hex / version_hint / bounds
#   container: TOC, gap/pad bytes, section order, original compressed LOD0
#   generic_meshes, terrain_patches, cloud, cloud_mesh   (legacy fields)
#   geo0: 0x38 global indices/draw groups, 0x39+ chunked terrain
#   metr: base64 of the 64-byte METR section
#
# Byte-exact round trip: original compressed meshopt buffers are stored and
# reused while the decoded data is unchanged; edited buffers are re-encoded
# with the embedded pure-Python meshopt vertex codec.
# ============================================================================
"""
Sky: Children of the Light — LVL/MESHES Decoder / Encoder / JSON Translator
=======================================================================
Menu features:
  1. meshes → JSON  (known layouts decoded; unknown bytes preserved explicitly)
  2. JSON → meshes  (binary roundtrip; runtime compatibility needs game testing)
  3. JSON translate (0x33..0x3D, strict / explicit geometry-compatibility mode)
  4. JSON → OBJ     (original exporter; modern layout limitations remain)
  5. OBJ import     (legacy terrain layouts only)
  6. MERGE two JSONs (original layout/version restrictions remain):
     - generic_meshes : keep first only
     - terrain_patches: concatenate with re-indexing
     - cloud           : **FULL volumetric merge** when dims + internal
                          block size match; otherwise safely keep first
     - cloud_mesh      : concatenate groups, keep first's mesh_16

  * Cloud merge decompresses all packed blocks using the *exact* LZ77
    algorithm (sub_100008B6C) with automatic size detection.
  * Merges per‑index‑entry dist/amb0/amb1 blocks, rebuilds sparse grid,
    index entries, and recompresses with the same algorithm.
  * Validation of file layout does not guarantee runtime/game resource compatibility.
  * No-argument startup exposes the Chinese interactive menu.
  * See meshes_JSON转译_使用说明.md for limitations and archive recovery.

Based on reverse-engineering of:
  - LevelData::LoadLod     (sub_10058F974)
  - CloudBarn::SetCloudData
  - BinaryStream           (sub_1007843F0 / ReadBits / WriteBits / WriteFloat3)
  - LZ77 decompressor      (sub_100008B6C)
"""

import struct
import sys
import os
import json
import math
import copy
import lz4.block


# ═══════════════════════════════════════════════════════════════════════════════
# Octahedron Normal Encoding (2× snorm16 in 4 bytes)
# ═══════════════════════════════════════════════════════════════════════════════

def _snorm16_to_float(v: int) -> float:
    """Decode a 16-bit signed-normalized integer to float [-1, 1]."""
    return max(-1.0, min(1.0, (v - 32767.5) / 32767.5))

def _float_to_snorm16(f: float) -> int:
    """Encode float [-1, 1] to 16-bit signed-normalized integer."""
    return max(0, min(65535, int(round(f * 32767.5 + 32767.5))))

def oct_decode_normal(packed: int):
    """Decode octahedron-encoded normal from packed uint32 (2× snorm16).
    Returns (nx, ny, nz) float triple. Terrain flat ground → (0, 0, 1)."""
    u = packed & 0xFFFF
    v = (packed >> 16) & 0xFFFF
    fx = _snorm16_to_float(u)
    fy = _snorm16_to_float(v)
    fz = 1.0 - abs(fx) - abs(fy)
    if fz < 0:
        # Octahedron wrap
        old_fx = fx
        fx = (1.0 - abs(fy)) * (-1.0 if old_fx < 0 else 1.0)
        fy = (1.0 - abs(old_fx)) * (-1.0 if fy < 0 else 1.0)
        fz = -fz
    # Normalize
    length = math.sqrt(fx * fx + fy * fy + fz * fz)
    if length > 0.0001:
        fx /= length
        fy /= length
        fz /= length
    return fx, fy, fz

def oct_encode_normal(nx: float, ny: float, nz: float) -> int:
    """Encode unit normal to octahedron packed uint32 (2× snorm16).
    Exact inverse of oct_decode_normal."""
    # Normalize input
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 0.0001:
        nx /= length
        ny /= length
        nz /= length
    # Project to octahedron
    denom = abs(nx) + abs(ny) + abs(nz)
    if denom < 0.0000001:
        return 0
    ox = nx / denom
    oy = ny / denom
    # Reflect for negative z
    if nz < 0:
        old_ox = ox
        ox = (1.0 - abs(oy)) * (1.0 if old_ox >= 0 else -1.0)
        oy = (1.0 - abs(old_ox)) * (1.0 if oy >= 0 else -1.0)
    # Map from [-1,1] to [0,1]
    ox = ox * 0.5 + 0.5
    oy = oy * 0.5 + 0.5
    # Quantize to snorm16
    u = _float_to_snorm16(ox * 2.0 - 1.0)
    v = _float_to_snorm16(oy * 2.0 - 1.0)
    return u | (v << 16)


# ═══════════════════════════════════════════════════════════════════════════════
# BinaryStream — bit/byte-accurate reader AND writer
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryStreamReader:
    """Reads from bytes, supporting byte-aligned AND bit-level reads."""
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_offset = 0

    def align(self):
        if self.bit_offset != 0:
            self.bit_offset = 0
            self.byte_pos += 1

    def remaining(self) -> int:
        return len(self.data) - self.byte_pos

    def _check(self, n: int):
        if self.byte_pos + n > len(self.data):
            raise EOFError(f"read at {self.byte_pos}+{n} > len={len(self.data)}")

    def read_bytes(self, n: int) -> bytes:
        self.align()
        if n <= 0:
            return b""
        self._check(n)
        res = self.data[self.byte_pos:self.byte_pos + n]
        self.byte_pos += n
        return res

    def read_uint32(self) -> int:
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_int32(self) -> int:
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_float(self) -> float:
        return struct.unpack('<f', self.read_bytes(4))[0]

    def read_float3(self) -> list:
        return list(struct.unpack('<3f', self.read_bytes(12)))

    def read_bits(self, count: int) -> int:
        value = 0
        bits_read = 0
        while bits_read < count:
            if self.byte_pos >= len(self.data):
                raise EOFError("read_bits past end")
            byte = self.data[self.byte_pos]
            remaining = 8 - self.bit_offset
            take = min(count - bits_read, remaining)
            mask = (1 << take) - 1
            chunk = (byte >> self.bit_offset) & mask
            value |= chunk << bits_read
            bits_read += take
            self.bit_offset += take
            if self.bit_offset >= 8:
                self.bit_offset = 0
                self.byte_pos += 1
        return value


class BinaryStreamWriter:
    """Writes bytes AND bits, producing the exact same format as the game."""
    def __init__(self):
        self.data = bytearray()
        self.bit_buf = 0
        self.bit_offset = 0

    def align(self):
        if self.bit_offset != 0:
            self.data.append(self.bit_buf & 0xFF)
            self.bit_buf = 0
            self.bit_offset = 0

    def write_bytes(self, b: bytes):
        self.align()
        self.data.extend(b)

    def write_uint32(self, v: int):
        self.write_bytes(struct.pack('<I', v & 0xFFFFFFFF))

    def write_int32(self, v: int):
        self.write_bytes(struct.pack('<i', v))

    def write_float(self, v: float):
        self.write_bytes(struct.pack('<f', v))

    def write_float3(self, vals: list):
        for v in vals[:3]:
            self.write_float(v)

    def write_bits(self, value: int, count: int):
        """Write `count` low bits of `value`. Matches BinaryStream::WriteBits."""
        if count <= 0:
            return
        if count > 32:
            raise ValueError(f"write_bits count={count} > 32")
        mask = (1 << count) - 1 if count < 32 else 0xFFFFFFFF
        v = value & mask
        written = 0
        while written < count:
            if self.bit_offset == 0:
                self.bit_buf = 0
            avail = 8 - self.bit_offset
            take = min(count - written, avail)
            chunk = (v >> written) & ((1 << take) - 1)
            self.bit_buf |= chunk << self.bit_offset
            self.bit_offset += take
            written += take
            if self.bit_offset >= 8:
                self.data.append(self.bit_buf & 0xFF)
                self.bit_offset = 0

    def to_bytes(self) -> bytes:
        self.align()
        return bytes(self.data)


# ═══════════════════════════════════════════════════════════════════════════════
# LVL0 Container — Decompress / Compress
# ═══════════════════════════════════════════════════════════════════════════════

def parse_toc(data: bytes) -> list:
    """Parse LVL/MESHES Table of Contents."""
    toc_start = 0x08
    entry_count = data[toc_start]
    entries = []
    for i in range(entry_count):
        base = toc_start + 4 + i * 12
        name = data[base:base + 4].rstrip(b'\x00').decode('ascii', errors='replace')
        offset = struct.unpack_from('<I', data, base + 4)[0]
        length = struct.unpack_from('<I', data, base + 8)[0]
        entries.append((name, offset, length))
    return entries


def decompress_lvl(input_path: str, output_dir: str) -> list:
    """Decompress LVL/MESHES file (LZ4 block compression)."""
    with open(input_path, 'rb') as f:
        data = f.read()

    if data[0:4] != b'LVL0':
        print(f"Magic mismatch: {data[0:4]}")
        return []

    version = struct.unpack_from('<I', data, 4)[0]
    print(f"LVL version: {version} (0x{version:x})")

    entries = parse_toc(data)
    print(f"TOC entries: {len(entries)}")
    for name, offset, length in entries:
        print(f"  {name}: offset=0x{offset:x} len=0x{length:x}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_paths = []

    for name, offset, length in entries:
        if offset + length > len(data):
            print(f"  {name}: skip (offset+len out of bounds)")
            continue
        compressed = data[offset:offset + length]
        try:
            decompressed = lz4.block.decompress(compressed, uncompressed_size=0xC00000)
            out_path = os.path.join(output_dir, f"{base_name}_{name}_decompressed.bin")
            with open(out_path, 'wb') as f:
                f.write(decompressed)
            print(f"  {name}: {len(compressed):,} -> {len(decompressed):,} bytes -> {out_path}")
            out_paths.append(out_path)
        except Exception as e:
            print(f"  {name}: LZ4 decompress failed: {e}")

    return out_paths


def compress_lvl(decompressed_paths: list, output_path: str, base_name: str = "LOD0", version: int = 0x36):
    """Re-compress decompressed bin(s) back into a LVL0 container."""
    toc_entries = []
    body_parts = []
    current_offset = 0

    for i, dpath in enumerate(decompressed_paths):
        with open(dpath, 'rb') as f:
            raw = f.read()
        compressed = lz4.block.compress(raw, store_size=False)
        entry_name = f"LOD{i}" if i == 0 else os.path.splitext(os.path.basename(dpath))[0].split('_')[1]
        entry_name = entry_name[:4].encode('ascii').ljust(4, b'\x00')
        toc_entries.append((entry_name, current_offset, len(compressed)))
        body_parts.append(compressed)
        current_offset += len(compressed)

    header = bytearray()
    header.extend(b'LVL0')
    header.extend(struct.pack('<I', version))
    header.append(len(toc_entries))
    header.extend(b'\xff\xff\xff')

    for name, offset, length in toc_entries:
        header.extend(name)
        header.extend(struct.pack('<I', offset))
        header.extend(struct.pack('<I', length))

    with open(output_path, 'wb') as f:
        f.write(bytes(header))
        for part in body_parts:
            f.write(part)

    print(f"LVL0 container written: {output_path} ({len(header) + sum(len(p) for p in body_parts)} bytes)")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# FULL DECODER — 100% field coverage
# ═══════════════════════════════════════════════════════════════════════════════

def decode_generic_meshes(stream: BinaryStreamReader) -> list:
    result = []
    total = stream.read_uint32()
    print(f"  Generic Meshes: {total} total")

    for i in range(total):
        mesh = {'index': i}
        name_len = stream.read_uint32()
        if name_len > 1024:
            print(f"  [ERROR] Mesh {i} name_len={name_len}, abort")
            stream.byte_pos -= 4
            break
        name = stream.read_bytes(name_len)
        mesh['name'] = name.decode('ascii', errors='replace')
        mesh['resource_id'] = stream.read_uint32()
        submesh_count = stream.read_uint32()
        mesh['submesh_count'] = submesh_count
        sep = stream.read_bytes(1)[0]
        mesh['separator'] = sep
        mesh['submeshes'] = []

        if sep == 0x00:
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            raw = stream.read_bytes(12 * cc)
            colors = []
            for b in range(cc):
                for q in range(3):
                    off = b * 12 + q * 4
                    colors.append([raw[off], raw[off+1], raw[off+2], raw[off+3]])
            gap_data = list(stream.read_bytes(gg))
            mesh['submeshes'].append({
                'mode': 0, 'color_blocks': cc, 'gap': gg,
                'colors': colors, 'gap_data': gap_data
            })
        elif sep == 0x01:
            mode = 1
            stream.read_bytes(4)
            gg = stream.read_uint32()
            raw = stream.read_bytes(12)
            colors = []
            for q in range(3):
                off = q * 4
                colors.append([raw[off], raw[off+1], raw[off+2], raw[off+3]])
            gap_data = list(stream.read_bytes(gg))
            mesh['submeshes'].append({
                'mode': 1, 'color_blocks': 1, 'gap': gg,
                'colors': colors, 'gap_data': gap_data
            })
        else:
            print(f"  [!] Mesh {i} {mesh['name']} unknown separator 0x{sep:02x}, treating as mode 0")
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            raw = stream.read_bytes(12 * cc)
            colors = []
            for b in range(cc):
                for q in range(3):
                    off = b * 12 + q * 4
                    colors.append([raw[off], raw[off+1], raw[off+2], raw[off+3]])
            gap_data = list(stream.read_bytes(gg))
            mesh['submeshes'].append({
                'mode': 0, 'color_blocks': cc, 'gap': gg,
                'colors': colors, 'gap_data': gap_data
            })

        for j in range(1, submesh_count):
            if mode == 0:
                cc = stream.read_uint32()
                gg = stream.read_uint32()
                raw = stream.read_bytes(12 * cc)
                colors = []
                for b in range(cc):
                    for q in range(3):
                        off = b * 12 + q * 4
                        colors.append([raw[off], raw[off+1], raw[off+2], raw[off+3]])
                gap_data = list(stream.read_bytes(gg))
                mesh['submeshes'].append({
                    'mode': 0, 'color_blocks': cc, 'gap': gg,
                    'colors': colors, 'gap_data': gap_data
                })
            elif mode == 1:
                stream.read_bytes(4)
                gg = stream.read_uint32()
                raw = stream.read_bytes(12)
                colors = []
                for q in range(3):
                    off = q * 4
                    colors.append([raw[off], raw[off+1], raw[off+2], raw[off+3]])
                gap_data = list(stream.read_bytes(gg))
                mesh['submeshes'].append({
                    'mode': 1, 'color_blocks': 1, 'gap': gg,
                    'colors': colors, 'gap_data': gap_data
                })

        result.append(mesh)

    print(f"  Generic Meshes parsed: {len(result)}, stream at {stream.byte_pos}")
    return result


def decode_terrain_patches(stream: BinaryStreamReader) -> list:
    patches = []
    patch_count = stream.read_uint32()
    print(f"\n  Terrain Patches: {patch_count}")

    if patch_count == 0 or patch_count > 100000:
        print("  Warning: patch_count anomalous, skipping")
        return patches

    for pi in range(patch_count):
        p = {'index': pi}
        p['external_id'] = stream.read_int32()
        p['flag1'] = bool(stream.read_bits(1))
        p['flag2'] = bool(stream.read_bits(1))
        stream.align()
        p['row0'] = stream.read_float3()
        p['row1'] = stream.read_float3()
        p['vert_count'] = stream.read_uint32()
        p['index_count'] = stream.read_uint32()

        if p['vert_count'] > 500000 or p['index_count'] > 1500000:
            print(f"  [ERROR] Patch {pi} vert={p['vert_count']} idx={p['index_count']}, abort terrain")
            break

        vert_bytes = stream.read_bytes(36 * p['vert_count'])
        vc = p['vert_count']
        px, py, pz = [0.0]*vc, [0.0]*vc, [0.0]*vc
        nx, ny, nz = [0.0]*vc, [0.0]*vc, [0.0]*vc
        norm_u, norm_v = [0]*vc, [0]*vc
        mat_ids = [0]*vc
        vert_attrs = [0]*vc
        cr, cg, cb, ca = [0]*vc, [0]*vc, [0]*vc, [0]*vc
        extra_a = [0]*vc
        extra_b = [0]*vc
        for v in range(vc):
            off = v * 36
            px[v], py[v], pz[v] = struct.unpack_from('<3f', vert_bytes, off)
            packed_normal = struct.unpack_from('<I', vert_bytes, off + 12)[0]
            nx[v], ny[v], nz[v] = oct_decode_normal(packed_normal)
            norm_u[v] = packed_normal & 0xFFFF
            norm_v[v] = (packed_normal >> 16) & 0xFFFF
            mat_ids[v] = struct.unpack_from('<I', vert_bytes, off + 16)[0]
            vert_attrs[v] = struct.unpack_from('<I', vert_bytes, off + 20)[0]
            cr[v], cg[v], cb[v], ca[v] = vert_bytes[off+24:off+28]
            extra_a[v] = struct.unpack_from('<I', vert_bytes, off + 28)[0]
            extra_b[v] = struct.unpack_from('<I', vert_bytes, off + 32)[0]
        p['v'] = {
            'px': px, 'py': py, 'pz': pz,
            'nx': nx, 'ny': ny, 'nz': nz,
            'normal_u': norm_u, 'normal_v': norm_v,
            'material_id': mat_ids, 'vertex_attr': vert_attrs,
            'cr': cr, 'cg': cg, 'cb': cb, 'ca': ca,
            'extra_a': extra_a, 'extra_b': extra_b
        }

        p['some_count'] = stream.read_uint32()
        if p['some_count'] > stream.remaining():
            print(f"  [ERROR] Patch {pi} some_count={p['some_count']} > remaining")
            break
        p['some_data'] = list(stream.read_bytes(p['some_count']))

        p['row2'] = stream.read_float3()
        p['row3'] = stream.read_float3()
        p['float_value'] = stream.read_float()
        p['uint_116'] = stream.read_uint32()
        p['uint_120'] = stream.read_uint32()
        p['uint_124'] = stream.read_uint32()
        p['count_A'] = stream.read_uint32()

        count_A = p['count_A']
        if count_A > 50000:
            print(f"  [ERROR] Patch {pi} count_A={count_A}")
            break

        elem_loops = []
        for _ in range(count_A):
            loop_val = stream.read_uint32()
            some_val = stream.read_uint32()
            elem_loops.append({'loop': loop_val, 'value': some_val})
        p['elem_loops'] = elem_loops

        p['count_A_repeat'] = stream.read_uint32()
        p['count_B'] = stream.read_uint32()
        p['count_C'] = stream.read_uint32()

        if p['count_B'] > stream.remaining() // 4 or p['count_C'] > stream.remaining() // 2:
            print(f"  [ERROR] Patch {pi} count_B={p['count_B']} count_C={p['count_C']} anomalous")
            break

        array_B = list(struct.unpack(f'<{p["count_B"]}f' if p['count_B'] > 0 else '',
                       stream.read_bytes(p['count_B'] * 4)))
        array_C = list(struct.unpack(f'<{p["count_C"]}H',
                       stream.read_bytes(p['count_C'] * 2)))
        array_A = [stream.read_uint32() for _ in range(p['count_A_repeat'])]

        p['array_B'] = array_B
        p['array_C'] = array_C
        p['array_A'] = array_A

        if p['count_B'] == 0 and p['index_count'] > 0:
            if p['index_count'] <= stream.remaining() // 2:
                raw_idx = list(struct.unpack(f'<{p["index_count"]}H',
                               stream.read_bytes(p['index_count'] * 2)))
                p['raw_indices'] = raw_idx
            else:
                p['raw_indices'] = []
        else:
            p['raw_indices'] = []

        p['vertex_positions'] = [[p['v']['px'][i], p['v']['py'][i], p['v']['pz'][i]] for i in range(vc)]

        p['triangles'] = []
        if p['count_B'] > 0 and count_A > 0:
            raw_b = struct.pack(f'<{p["count_B"]}f', *array_B)
            indices = []
            idx_c = 0
            base = 0
            for ii in range(count_A):
                loop = elem_loops[ii]['loop']
                if loop == 0:
                    base += array_A[ii] if ii < len(array_A) else 0
                    continue
                for _ in range(loop):
                    if idx_c >= len(array_C):
                        break
                    code = array_C[idx_c]
                    if code & 1:
                        off = 4 * base + 2 + 4 * (code >> 1)
                    else:
                        off = 4 * base + 4 * (code >> 1)
                    if off + 2 <= len(raw_b):
                        val = struct.unpack_from('<H', raw_b, off)[0]
                        indices.append(val)
                    idx_c += 1
                base += array_A[ii] if ii < len(array_A) else 0

            for t in range(0, len(indices) - 2, 3):
                i1, i2, i3 = indices[t], indices[t+1], indices[t+2]
                if i1 == i2 or i2 == i3 or i1 == i3:
                    continue
                if max(i1, i2, i3) >= p['vert_count']:
                    continue
                p['triangles'].append([i1, i2, i3])
        elif p['raw_indices']:
            indices = p['raw_indices']
            for t in range(0, len(indices) - 2, 3):
                i1, i2, i3 = indices[t], indices[t+1], indices[t+2]
                if i1 == i2 or i2 == i3 or i1 == i3:
                    continue
                if max(i1, i2, i3) >= p['vert_count']:
                    continue
                p['triangles'].append([i1, i2, i3])

        patches.append(p)

        if pi < 3 or pi == patch_count - 1:
            print(f"    Patch {pi}: ext_id={p['external_id']} vert={p['vert_count']} "
                  f"idx={p['index_count']} "
                  f"cA={p['count_A']} cAr={p['count_A_repeat']} cB={p['count_B']} cC={p['count_C']} "
                  f"tris={len(p['triangles'])}")

    print(f"  Terrain parsed: {len(patches)} patches, stream at {stream.byte_pos}")
    return patches


def decode_cloud_section(stream: BinaryStreamReader) -> dict:
    stream.align()
    cloud_flag = stream.read_uint32()
    print(f"\n  Cloud flag: {cloud_flag} (0x{cloud_flag:08x})")
    if cloud_flag == 0 or cloud_flag == 0xFFFFFFFF:
        print("  No cloud data")
        return {'present': False}

    result = {'present': True}
    try:
        hdr_raw = [stream.read_uint32() for _ in range(6)]
        hdr_signed = [struct.unpack('<i', struct.pack('<I', x))[0] for x in hdr_raw]
        result['header'] = hdr_raw
        result['header_signed'] = hdr_signed
        result['bbox_origin'] = hdr_signed[0:3]
        result['dim'] = hdr_raw[3:6]
        dim_x, dim_y, dim_z = hdr_raw[3], hdr_raw[4], hdr_raw[5]
        print(f"  Cloud header: bbox_origin={hdr_signed[0:3]}, dim={dim_x}x{dim_y}x{dim_z}")

        if dim_x == 0 or dim_y == 0 or dim_z == 0 or dim_x > 0x10000:
            print(f"  Invalid cloud dims: {dim_x}x{dim_y}x{dim_z}")
            result['raw_remaining'] = stream.read_bytes(stream.remaining()).hex()
            return result

        sparse_size = dim_x * dim_y * dim_z
        if sparse_size > stream.remaining():
            print(f"  Sparse grid size {sparse_size} > remaining {stream.remaining()}")
            result['raw_remaining'] = stream.read_bytes(stream.remaining()).hex()
            return result

        sparse_raw = stream.read_bytes(sparse_size)
        non_empty = []
        for idx in range(sparse_size):
            if sparse_raw[idx] != 0xFF:
                z = idx // (dim_x * dim_y)
                y = (idx // dim_x) % dim_y
                x = idx % dim_x
                non_empty.append([x, y, z, sparse_raw[idx]])
        result['sparse_non_empty'] = non_empty
        result['sparse_total'] = sparse_size
        print(f"  Sparse grid: {sparse_size} cells ({dim_x}x{dim_y}x{dim_z}), {len(non_empty)} non-empty")

        if stream.remaining() < 4:
            return result
        count1 = stream.read_uint32()
        result['count1'] = count1
        print(f"  count1 = {count1}")

        if count1 > 0 and count1 * 6 <= stream.remaining():
            idx_raw = stream.read_bytes(count1 * 6)
            result['index_entries'] = []
            for c in range(count1):
                off = c * 6
                x = struct.unpack_from('<h', idx_raw, off)[0]
                y = struct.unpack_from('<h', idx_raw, off + 2)[0]
                z = struct.unpack_from('<h', idx_raw, off + 4)[0]
                result['index_entries'].append([x, y, z])
        else:
            result['index_entries'] = []

        if stream.remaining() < 12:
            return result
        v3 = [stream.read_uint32() for _ in range(3)]
        result['v3_sizes'] = v3
        print(f"  v3 (packed sizes): {v3}")

        v3_names = ['packed_dist', 'packed_amb0', 'packed_amb1']
        for idx, sz in enumerate(v3):
            if 0 < sz <= stream.remaining():
                result[v3_names[idx]] = list(stream.read_bytes(sz))
                print(f"    {v3_names[idx]}: {sz} bytes")
            elif sz > 0:
                print(f"    [skip] {v3_names[idx]}: {sz} > remaining {stream.remaining()}")
                break

        if stream.remaining() >= 8:
            result['float_v5'] = stream.read_float()
            result['v5_single'] = stream.read_uint32()
            print(f"  float_v5={result['float_v5']:.4f} v5_single={result['v5_single']}")

        if stream.remaining() >= 12:
            v6 = [stream.read_uint32() for _ in range(3)]
            result['v6'] = v6
            print(f"  v6: {v6}")

            if 0 < v6[1] <= stream.remaining() // 16:
                v6r = stream.read_bytes(16 * v6[1])
                v6c = v6[1]
                v6px, v6py, v6pz = [0.0]*v6c, [0.0]*v6c, [0.0]*v6c
                v6nx, v6ny, v6nz = [0.0]*v6c, [0.0]*v6c, [0.0]*v6c
                v6nu, v6nv = [0]*v6c, [0]*v6c
                for r in range(v6c):
                    off = r * 16
                    v6px[r], v6py[r], v6pz[r] = struct.unpack_from('<3f', v6r, off)
                    packed = struct.unpack_from('<I', v6r, off + 12)[0]
                    v6nx[r], v6ny[r], v6nz[r] = oct_decode_normal(packed)
                    v6nu[r] = packed & 0xFFFF
                    v6nv[r] = (packed >> 16) & 0xFFFF
                result['v6_v'] = {'px': v6px, 'py': v6py, 'pz': v6pz,
                                  'nx': v6nx, 'ny': v6ny, 'nz': v6nz,
                                  'normal_u': v6nu, 'normal_v': v6nv}
                print(f"    v6_vertices: {v6c} records (16 bytes each)")
            else:
                result['v6_v'] = {'px': [], 'py': [], 'pz': [],
                                  'nx': [], 'ny': [], 'nz': [],
                                  'normal_u': [], 'normal_v': []}

            if 0 < v6[2] <= stream.remaining() // 2:
                v6i = stream.read_bytes(2 * v6[2])
                result['v6_indices'] = list(struct.unpack(f'<{v6[2]}H', v6i))
                print(f"    v6_indices: {v6[2]} uint16")
            else:
                result['v6_indices'] = []

        return result

    except EOFError as e:
        print(f"  Cloud parse EOF: {e}")
        result['raw_remaining'] = stream.read_bytes(stream.remaining()).hex()
        return result


def decode_cloud_mesh(stream: BinaryStreamReader) -> dict:
    result = {'groups': [], 'mesh_16': None}
    if stream.remaining() < 5:
        return result

    pos_before = stream.byte_pos
    try:
        flag = stream.read_bytes(1)[0]
        if flag != 0x01:
            stream.byte_pos = pos_before
            return result

        group_count = stream.read_uint32()
        print(f"\n  CloudMesh groups: flag=0x{flag:02x}, count={group_count}")
        if group_count == 0 or group_count > 50000:
            stream.byte_pos = pos_before
            return result

        for gi in range(group_count):
            if stream.remaining() < 8:
                break
            vert_count = stream.read_uint32()
            if vert_count == 0 or vert_count > stream.remaining() // 40:
                stream.byte_pos -= 4
                break
            vert_raw = stream.read_bytes(40 * vert_count)
            vc_g = vert_count
            gpx, gpy, gpz = [0.0]*vc_g, [0.0]*vc_g, [0.0]*vc_g
            gnx, gny, gnz = [0.0]*vc_g, [0.0]*vc_g, [0.0]*vc_g
            gnu, gnv = [0]*vc_g, [0]*vc_g
            gmat, gattr = [0]*vc_g, [0]*vc_g
            gcr, gcg, gcb, gca = [0]*vc_g, [0]*vc_g, [0]*vc_g, [0]*vc_g
            gexa, gexb, gexc = [0]*vc_g, [0]*vc_g, [0]*vc_g
            for v in range(vc_g):
                off = v * 40
                gpx[v], gpy[v], gpz[v] = struct.unpack_from('<3f', vert_raw, off)
                packed = struct.unpack_from('<I', vert_raw, off + 12)[0]
                gnx[v], gny[v], gnz[v] = oct_decode_normal(packed)
                gnu[v] = packed & 0xFFFF
                gnv[v] = (packed >> 16) & 0xFFFF
                gmat[v] = struct.unpack_from('<I', vert_raw, off + 16)[0]
                gattr[v] = struct.unpack_from('<I', vert_raw, off + 20)[0]
                gcr[v], gcg[v], gcb[v], gca[v] = vert_raw[off + 24:off + 28]
                gexa[v] = struct.unpack_from('<I', vert_raw, off + 28)[0]
                gexb[v] = struct.unpack_from('<I', vert_raw, off + 32)[0]
                gexc[v] = struct.unpack_from('<I', vert_raw, off + 36)[0]

            if stream.remaining() < 4:
                break
            idx_count = stream.read_uint32()
            if idx_count > stream.remaining() // 2:
                break
            idx_raw = stream.read_bytes(2 * idx_count)
            indices = list(struct.unpack(f'<{idx_count}H', idx_raw))

            triangles = []
            for t in range(0, len(indices) - 2, 3):
                i1, i2, i3 = indices[t], indices[t+1], indices[t+2]
                if i1 == i2 or i2 == i3 or i1 == i3:
                    continue
                if max(i1, i2, i3) >= vert_count:
                    continue
                triangles.append([i1, i2, i3])

            result['groups'].append({
                'vert_count': vert_count,
                'v': {'px': gpx, 'py': gpy, 'pz': gpz,
                      'nx': gnx, 'ny': gny, 'nz': gnz,
                      'normal_u': gnu, 'normal_v': gnv,
                      'material_id': gmat, 'vertex_attr': gattr,
                      'cr': gcr, 'cg': gcg, 'cb': gcb, 'ca': gca,
                      'extra_a': gexa, 'extra_b': gexb, 'extra_c': gexc},
                'idx_count': idx_count,
                'indices': indices,
                'triangles': triangles,
            })

        if stream.remaining() >= 12:
            N = stream.read_uint32()
            print(f"  CloudMesh 16-byte section: N={N}")
            if 0 < N <= 100:
                if stream.remaining() >= 8:
                    a = stream.read_uint32()
                    b = stream.read_uint32()
                    if 0 < a <= stream.remaining() // 16 and 0 < b <= stream.remaining() // 2:
                        vert_r16 = stream.read_bytes(16 * a)
                        p16x, p16y, p16z = [0.0]*a, [0.0]*a, [0.0]*a
                        p16nx, p16ny, p16nz = [0.0]*a, [0.0]*a, [0.0]*a
                        p16nu, p16nv = [0]*a, [0]*a
                        for v in range(a):
                            off = v * 16
                            p16x[v], p16y[v], p16z[v] = struct.unpack_from('<3f', vert_r16, off)
                            packed = struct.unpack_from('<I', vert_r16, off + 12)[0]
                            p16nx[v], p16ny[v], p16nz[v] = oct_decode_normal(packed)
                            p16nu[v] = packed & 0xFFFF
                            p16nv[v] = (packed >> 16) & 0xFFFF

                        idx_r16 = stream.read_bytes(2 * b)
                        indices_16 = list(struct.unpack(f'<{b}H', idx_r16))

                        positions_16 = [[p16x[i], p16y[i], p16z[i]] for i in range(a)]
                        triangles_16 = []
                        for t in range(0, len(indices_16) - 2, 3):
                            i1, i2, i3 = indices_16[t], indices_16[t+1], indices_16[t+2]
                            if i1 == i2 or i2 == i3 or i1 == i3:
                                continue
                            if max(i1, i2, i3) >= a:
                                continue
                            triangles_16.append([i1, i2, i3])

                        result['mesh_16'] = {
                            'vert_count': a,
                            'v': {'px': p16x, 'py': p16y, 'pz': p16z,
                                  'nx': p16nx, 'ny': p16ny, 'nz': p16nz,
                                  'normal_u': p16nu, 'normal_v': p16nv},
                            'positions': positions_16,
                            'idx_count': b,
                            'indices': indices_16,
                            'triangles': triangles_16,
                        }
        return result

    except Exception as e:
        print(f"  CloudMesh parse error: {e}")
        stream.byte_pos = pos_before
        return {'groups': [], 'mesh_16': None}


def decode_meshes_full(stream: BinaryStreamReader) -> dict:
    result = {}
    result['generic_meshes'] = decode_generic_meshes(stream)
    result['terrain_patches'] = decode_terrain_patches(stream)
    result['cloud'] = decode_cloud_section(stream)
    result['cloud_mesh'] = decode_cloud_mesh(stream)
    remaining = stream.remaining()
    if remaining > 0:
        print(f"\n  Remaining unknown data: {remaining} bytes")
        result['_remaining'] = list(stream.read_bytes(remaining))
    print(f"\nDecode complete. Stream at {stream.byte_pos}, remaining {stream.remaining()}")
    return result


def meshes_to_json(input_path: str, output_path: str):
    with open(input_path, 'rb') as f:
        data = f.read()

    if data[:4] == b'LVL0':
        print("Detected LVL0 container, decompressing first...")
        entries = parse_toc(data)
        if not entries:
            print("No TOC entries found")
            return
        name, offset, length = entries[0]
        compressed = data[offset:offset + length]
        try:
            raw = lz4.block.decompress(compressed, uncompressed_size=0xC00000)
        except Exception as e:
            print(f"LZ4 decompress failed: {e}")
            return
    else:
        raw = data

    stream = BinaryStreamReader(raw)
    decoded = decode_meshes_full(stream)
    decoded['_meta'] = {
        'source': os.path.basename(input_path),
        'total_bytes': len(raw),
        'parsed_bytes': stream.byte_pos,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decoded, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {output_path}")
    return decoded


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ENCODER — JSON → Binary (roundtrip)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_generic_meshes(writer: BinaryStreamWriter, meshes: list):
    writer.write_uint32(len(meshes))
    for m in meshes:
        name_bytes = m['name'].encode('ascii')
        writer.write_uint32(len(name_bytes))
        writer.write_bytes(name_bytes)
        writer.write_uint32(m['resource_id'])
        writer.write_uint32(m['submesh_count'])
        sep = m['separator']
        writer.write_bytes(bytes([sep]))
        mode = m['submeshes'][0]['mode']
        for j, sm in enumerate(m['submeshes']):
            if j == 0:
                if mode == 0:
                    writer.write_uint32(sm['color_blocks'])
                    writer.write_uint32(sm['gap'])
                    raw = bytearray()
                    for b in range(sm['color_blocks']):
                        for q in range(3):
                            c = sm['colors'][b * 3 + q]
                            raw.extend(bytes(c))
                    writer.write_bytes(bytes(raw))
                    writer.write_bytes(bytes(sm['gap_data']))
                else:
                    writer.write_bytes(b'\x00\x00\x00\x00')
                    writer.write_uint32(sm['gap'])
                    raw = bytearray()
                    for q in range(3):
                        raw.extend(bytes(sm['colors'][q]))
                    writer.write_bytes(bytes(raw))
                    writer.write_bytes(bytes(sm['gap_data']))
            else:
                if mode == 0:
                    writer.write_uint32(sm['color_blocks'])
                    writer.write_uint32(sm['gap'])
                    raw = bytearray()
                    for b in range(sm['color_blocks']):
                        for q in range(3):
                            c = sm['colors'][b * 3 + q]
                            raw.extend(bytes(c))
                    writer.write_bytes(bytes(raw))
                    writer.write_bytes(bytes(sm['gap_data']))
                else:
                    writer.write_bytes(b'\x00\x00\x00\x00')
                    writer.write_uint32(sm['gap'])
                    raw = bytearray()
                    for q in range(3):
                        raw.extend(bytes(sm['colors'][q]))
                    writer.write_bytes(bytes(raw))
                    writer.write_bytes(bytes(sm['gap_data']))


def encode_terrain_patches(writer: BinaryStreamWriter, patches: list):
    writer.write_uint32(len(patches))
    for p in patches:
        writer.write_int32(p['external_id'])
        writer.write_bits(1 if p['flag1'] else 0, 1)
        writer.write_bits(1 if p['flag2'] else 0, 1)
        writer.align()
        writer.write_float3(p['row0'])
        writer.write_float3(p['row1'])
        writer.write_uint32(p['vert_count'])
        writer.write_uint32(p['index_count'])

        vdat = p['v']
        vc = p['vert_count']
        vert_data = bytearray()
        for i in range(vc):
            vert_data.extend(struct.pack('<3f', vdat['px'][i], vdat['py'][i], vdat['pz'][i]))
            packed_normal = vdat['normal_u'][i] | (vdat['normal_v'][i] << 16)
            vert_data.extend(struct.pack('<I', packed_normal))
            vert_data.extend(struct.pack('<I', vdat['material_id'][i]))
            vert_data.extend(struct.pack('<I', vdat['vertex_attr'][i]))
            vert_data.append(vdat['cr'][i])
            vert_data.append(vdat['cg'][i])
            vert_data.append(vdat['cb'][i])
            vert_data.append(vdat['ca'][i])
            vert_data.extend(struct.pack('<I', vdat['extra_a'][i]))
            vert_data.extend(struct.pack('<I', vdat['extra_b'][i]))
        writer.write_bytes(bytes(vert_data))

        writer.write_uint32(p['some_count'])
        writer.write_bytes(bytes(p['some_data']))
        writer.write_float3(p['row2'])
        writer.write_float3(p['row3'])
        writer.write_float(p['float_value'])
        writer.write_uint32(p['uint_116'])
        writer.write_uint32(p['uint_120'])
        writer.write_uint32(p['uint_124'])

        writer.write_uint32(p['count_A'])
        for el in p['elem_loops']:
            writer.write_uint32(el['loop'])
            writer.write_uint32(el['value'])

        writer.write_uint32(p.get('count_A_repeat', p['count_A']))
        writer.write_uint32(p['count_B'])
        writer.write_uint32(p['count_C'])

        for v in p['array_B']:
            writer.write_float(v)
        for v in p['array_C']:
            writer.write_bytes(struct.pack('<H', v))
        for v in p['array_A']:
            writer.write_uint32(v)

        if p['count_B'] == 0 and p.get('raw_indices'):
            for v in p['raw_indices']:
                writer.write_bytes(struct.pack('<H', v))


def encode_cloud_section(writer: BinaryStreamWriter, cloud: dict):
    if not cloud.get('present'):
        writer.write_uint32(0)
        return
    writer.write_uint32(1)
    bbox_origin = cloud.get('bbox_origin', cloud.get('header_signed', [0, 0, 0])[:3])
    dims = cloud.get('dim', [0, 0, 0])
    for v in bbox_origin:
        writer.write_uint32(v & 0xFFFFFFFF)
    for v in dims:
        writer.write_uint32(v)

    sparse_total = cloud.get('sparse_total', dims[0] * dims[1] * dims[2])
    sparse = bytearray([0xFF]) * sparse_total
    for entry in cloud.get('sparse_non_empty', []):
        x, y, z, v = entry
        idx = z * dims[0] * dims[1] + y * dims[0] + x
        if idx < len(sparse):
            sparse[idx] = v
    writer.write_bytes(bytes(sparse))

    writer.write_uint32(cloud['count1'])
    if cloud.get('index_entries'):
        for entry in cloud['index_entries']:
            writer.write_bytes(struct.pack('<3h', *entry))

    for v in cloud.get('v3_sizes', [0, 0, 0]):
        writer.write_uint32(v)

    for key in ['packed_dist', 'packed_amb0', 'packed_amb1']:
        if key in cloud and cloud[key]:
            writer.write_bytes(bytes(cloud[key]))

    if 'float_v5' in cloud:
        writer.write_float(cloud['float_v5'])
    if 'v5_single' in cloud:
        writer.write_uint32(cloud['v5_single'])

    v6 = cloud.get('v6', [0, 0, 0])
    for v in v6:
        writer.write_uint32(v)
    if cloud.get('v6_v'):
        v6v = cloud['v6_v']
        for i in range(len(v6v['px'])):
            writer.write_bytes(struct.pack('<3f', v6v['px'][i], v6v['py'][i], v6v['pz'][i]))
            writer.write_uint32(v6v['normal_u'][i] | (v6v['normal_v'][i] << 16))
    if cloud.get('v6_indices'):
        for v in cloud['v6_indices']:
            writer.write_bytes(struct.pack('<H', v))


def encode_cloud_mesh(writer: BinaryStreamWriter, cloud_mesh: dict):
    groups = cloud_mesh.get('groups', [])
    mesh_16 = cloud_mesh.get('mesh_16')
    if not groups and not mesh_16:
        writer.write_bytes(b'\x00')
        return

    writer.write_bytes(b'\x01')
    writer.write_uint32(len(groups))
    for g in groups:
        writer.write_uint32(g['vert_count'])
        vcg = g['vert_count']
        gv = g['v']
        for i in range(vcg):
            writer.write_bytes(struct.pack('<3f', gv['px'][i], gv['py'][i], gv['pz'][i]))
            writer.write_uint32(gv['normal_u'][i] | (gv['normal_v'][i] << 16))
            writer.write_uint32(gv['material_id'][i])
            writer.write_uint32(gv['vertex_attr'][i])
            writer.write_bytes(bytes([gv['cr'][i], gv['cg'][i], gv['cb'][i], gv['ca'][i]]))
            writer.write_uint32(gv['extra_a'][i])
            writer.write_uint32(gv['extra_b'][i])
            writer.write_uint32(gv['extra_c'][i])
        writer.write_uint32(g['idx_count'])
        for v in g['indices']:
            writer.write_bytes(struct.pack('<H', v))

    if mesh_16:
        writer.write_uint32(1)
        writer.write_uint32(mesh_16['vert_count'])
        writer.write_uint32(mesh_16['idx_count'])
        mv = mesh_16['v']
        for i in range(mesh_16['vert_count']):
            writer.write_bytes(struct.pack('<3f', mv['px'][i], mv['py'][i], mv['pz'][i]))
            writer.write_uint32(mv['normal_u'][i] | (mv['normal_v'][i] << 16))
        for v in mesh_16['indices']:
            writer.write_bytes(struct.pack('<H', v))
    else:
        writer.write_uint32(0)


def encode_meshes_full(decoded: dict) -> bytes:
    writer = BinaryStreamWriter()
    encode_generic_meshes(writer, decoded.get('generic_meshes', []))
    encode_terrain_patches(writer, decoded.get('terrain_patches', []))
    encode_cloud_section(writer, decoded.get('cloud', {'present': False}))
    encode_cloud_mesh(writer, decoded.get('cloud_mesh', {}))
    if decoded.get('_remaining'):
        writer.write_bytes(bytes(decoded['_remaining']))
    return writer.to_bytes()


def _compute_terrain_bbox(decoded: dict) -> tuple:
    patches = decoded.get('terrain_patches', [])
    if not patches:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for p in patches:
        v = p.get('v', {})
        px = v.get('px', [])
        py = v.get('py', [])
        pz = v.get('pz', [])
        for i in range(len(px)):
            min_x = min(min_x, px[i])
            min_y = min(min_y, py[i])
            min_z = min(min_z, pz[i])
            max_x = max(max_x, px[i])
            max_y = max(max_y, py[i])
            max_z = max(max_z, pz[i])
    if min_x == float('inf'):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def json_to_meshes(json_path: str, output_path: str, as_lvl: bool = True):
    with open(json_path, 'r', encoding='utf-8') as f:
        decoded = json.load(f)

    raw = encode_meshes_full(decoded)
    print(f"Encoded binary: {len(raw)} bytes")

    if as_lvl:
        compressed = lz4.block.compress(raw, store_size=False)
        print(f"LZ4 compressed: {len(compressed)} bytes")
        LVL_HEADER_SIZE = 0x88
        header = bytearray(LVL_HEADER_SIZE)
        header[0:4] = b'LVL0'
        struct.pack_into('<I', header, 4, 0x36)
        header[8] = 1
        header[9:12] = b'\xff\xff\xff'
        header[12:16] = b'LOD0'
        struct.pack_into('<I', header, 0x10, LVL_HEADER_SIZE)
        struct.pack_into('<I', header, 0x14, len(compressed))

        bbox = _compute_terrain_bbox(decoded)
        struct.pack_into('<6f', header, 0x70, *bbox)
        print(f"Terrain bbox: min=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f}) "
              f"max=({bbox[3]:.1f},{bbox[4]:.1f},{bbox[5]:.1f})")

        full = bytes(header) + compressed
        with open(output_path, 'wb') as f:
            f.write(full)
        print(f"LVL0 meshes written: {output_path} ({len(full)} bytes)")
    else:
        with open(output_path, 'wb') as f:
            f.write(raw)
        print(f"Raw binary written: {output_path} ({len(raw)} bytes)")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# OBJ IMPORT / EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_obj(obj_path: str) -> tuple:
    vertices = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                indices = []
                for p in parts[1:4]:
                    idx = int(p.split('/')[0]) - 1
                    indices.append(idx)
                faces.append(indices)
    print(f"OBJ parsed: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def import_obj_to_json(obj_path: str, json_path: str, output_json_path: str = None,
                       external_id: int = -1):
    if output_json_path is None:
        output_json_path = json_path

    with open(json_path, 'r', encoding='utf-8') as f:
        decoded = json.load(f)

    obj_vertices, obj_faces = parse_obj(obj_path)
    if not obj_vertices or not obj_faces:
        print("OBJ 无几何体，操作终止")
        return None

    existing_patches = decoded.get('terrain_patches', [])
    mat_counter = {}
    for p in existing_patches:
        mats = p.get('v', {}).get('material_id', [])
        for m in mats:
            if m != 0:
                mat_counter[m] = mat_counter.get(m, 0) + 1

    if not mat_counter:
        print("错误：关卡中没有任何顶点使用非0材质ID，无法导入OBJ。")
        return None

    default_mat_id = max(mat_counter, key=mat_counter.get)
    print(f"选择最常见材质ID: {default_mat_id} (出现 {mat_counter[default_mat_id]} 次)")

    if existing_patches:
        template = copy.deepcopy(existing_patches[-1])
        print(f"复用 terrain patch #{template['index']} 的元数据")
    else:
        template = {
            'row0': [1.0, 0.0, 0.0], 'row1': [0.0, 1.0, 0.0],
            'row2': [0.0, 0.0, 1.0], 'row3': [0.0, 0.0, 0.0],
            'float_value': 1.0, 'uint_116': 0, 'uint_120': 0, 'uint_124': 0,
            'some_count': 0, 'some_data': [],
            'count_A': 0, 'elem_loops': [],
            'count_A_repeat': 0, 'count_B': 0, 'count_C': 0,
            'array_B': [], 'array_C': [], 'array_A': []
        }

    vert_count = len(obj_vertices)
    index_count = len(obj_faces) * 3
    packed_normal = oct_encode_normal(0.0, 0.0, 1.0)
    DEFAULT_NU = packed_normal & 0xFFFF
    DEFAULT_NV = (packed_normal >> 16) & 0xFFFF

    vdat = {
        'px': [v[0] for v in obj_vertices],
        'py': [v[1] for v in obj_vertices],
        'pz': [v[2] for v in obj_vertices],
        'nx': [0.0]*vert_count, 'ny': [0.0]*vert_count, 'nz': [1.0]*vert_count,
        'normal_u': [DEFAULT_NU]*vert_count,
        'normal_v': [DEFAULT_NV]*vert_count,
        'material_id': [default_mat_id] * vert_count,
        'vertex_attr': [255] * vert_count,
        'cr': [255]*vert_count, 'cg': [255]*vert_count,
        'cb': [255]*vert_count, 'ca': [255]*vert_count,
        'extra_a': [0]*vert_count, 'extra_b': [0]*vert_count
    }

    raw_indices = [idx for face in obj_faces for idx in face]

    new_patch = {
        'index': len(existing_patches),
        'external_id': external_id,
        'flag1': False, 'flag2': False,
        'row0': template['row0'], 'row1': template['row1'],
        'row2': template['row2'], 'row3': template['row3'],
        'float_value': template['float_value'],
        'uint_116': template['uint_116'],
        'uint_120': template['uint_120'],
        'uint_124': template['uint_124'],
        'some_count': template['some_count'],
        'some_data': template['some_data'],
        'count_A': template['count_A'],
        'elem_loops': template['elem_loops'],
        'count_A_repeat': template['count_A_repeat'],
        'count_B': template['count_B'],
        'count_C': template['count_C'],
        'array_B': template['array_B'],
        'array_C': template['array_C'],
        'array_A': template['array_A'],
        'vert_count': vert_count,
        'index_count': index_count,
        'v': vdat,
        'raw_indices': raw_indices,
        'vertex_positions': obj_vertices,
        'triangles': obj_faces,
    }

    if 'terrain_patches' not in decoded:
        decoded['terrain_patches'] = []
    decoded['terrain_patches'].append(new_patch)

    if '_remaining' in decoded:
        del decoded['_remaining']

    decoded['_meta']['modified'] = True
    decoded['_meta']['obj_import'] = os.path.basename(obj_path)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(decoded, f, indent=2, ensure_ascii=False)

    print(f"OBJ 已作为 terrain patch {new_patch['index']} 导入")
    print(f"  Vertices: {vert_count}, Triangles: {len(obj_faces)}")
    print(f"  材质ID: {default_mat_id} (来源于 {mat_counter[default_mat_id]} 个顶点样本)")
    print(f"  输出: {output_json_path}")
    return decoded


def json_to_obj(json_path: str, output_dir: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        decoded = json.load(f)

    base = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    all_verts = []
    all_tris = []
    vert_offset = 0

    for pi, p in enumerate(decoded.get('terrain_patches', [])):
        positions = p.get('vertex_positions', [])
        triangles = p.get('triangles', [])
        if not positions and p.get('vertices'):
            positions = [v['pos'] for v in p['vertices']]
        if positions and triangles:
            for v in positions:
                all_verts.append(v)
            for tri in triangles:
                all_tris.append([tri[0] + vert_offset, tri[1] + vert_offset, tri[2] + vert_offset])
            vert_offset += len(positions)

    for gi, g in enumerate(decoded.get('cloud_mesh', {}).get('groups', [])):
        gv = g.get('v', {})
        vc = g.get('vert_count', 0)
        triangles = g.get('triangles', [])
        if vc and triangles and gv:
            for i in range(vc):
                all_verts.append([gv['px'][i], gv['py'][i], gv['pz'][i]])
            for tri in triangles:
                all_tris.append([tri[0] + vert_offset, tri[1] + vert_offset, tri[2] + vert_offset])
            vert_offset += vc

    mesh_16 = decoded.get('cloud_mesh', {}).get('mesh_16')
    if mesh_16:
        mv = mesh_16.get('v', {})
        mvc = mesh_16.get('vert_count', 0)
        triangles = mesh_16.get('triangles', [])
        if mvc and triangles and mv:
            for i in range(mvc):
                all_verts.append([mv['px'][i], mv['py'][i], mv['pz'][i]])
            for tri in triangles:
                all_tris.append([tri[0] + vert_offset, tri[1] + vert_offset, tri[2] + vert_offset])
            vert_offset += mvc

    obj_path = os.path.join(output_dir, f"{base}.obj")
    with open(obj_path, 'w') as f:
        f.write(f"# {base} - terrain + cloud meshes\n")
        f.write(f"# {len(all_verts)} vertices, {len(all_tris)} triangles\n")
        for v in all_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in all_tris:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"OBJ exported: {obj_path} ({len(all_verts)}v, {len(all_tris)}t)")
    return obj_path


# ═══════════════════════════════════════════════════════════════════════════════
# LZ77 decompressor / compressor (exact match with sub_100008B6C)
# ═══════════════════════════════════════════════════════════════════════════════

_QWORD_10083E390 = [4, 1, 2, 1, 4, 4, 4, 4]
_QWORD_10083E3D0 = [0, 0, 0, -1, 0, 1, 2, 3]

def _lz77_decompress_auto(src: bytes) -> bytearray:
    """Decompress LZ77 stream without needing the decompressed size."""
    src = bytearray(src)
    dst = bytearray()
    src_idx = 0
    src_len = len(src)

    def expand_length(initial, idx):
        length = initial
        while idx < src_len:
            ext = src[idx]; idx += 1
            length += ext
            if ext != 255:
                break
        return length, idx

    while src_idx < src_len:
        if src_idx >= src_len:
            break
        token = src[src_idx]; src_idx += 1
        length = token >> 4
        dist_low = token & 0xF

        if length == 15:
            length, src_idx = expand_length(0, src_idx)

        if src_idx + length > src_len:
            dst.extend(src[src_idx:])
            break
        dst.extend(src[src_idx:src_idx+length])
        src_idx += length

        if src_idx >= src_len:
            break

        distance = dist_low
        if dist_low == 15:
            distance, src_idx = expand_length(0, src_idx)

        match_len = length + 4
        match_offset = len(dst) - distance - 1

        diff = len(dst) - match_offset
        if diff <= 7 and diff > 0:
            base = match_offset + _QWORD_10083E3D0[diff]
            for _ in range(match_len):
                dst.append(dst[base]); base += 1
        else:
            for i in range(match_len):
                dst.append(dst[match_offset + i])

    return dst


def _lz77_compress(data: bytes) -> bytes:
    """Compress producing game-compatible LZ77 stream."""
    MAX_DIST = 32768
    MIN_MATCH = 4
    out = bytearray()
    i = 0
    n = len(data)

    while i < n:
        best_len = 0
        best_dist = 0
        max_dist = min(i, MAX_DIST)
        for dist in range(1, max_dist + 1):
            j = i - dist
            k = i
            while k < n and data[k] == data[j] and (k - i) < 271:
                k += 1
                j += 1
            match_len = k - i
            if match_len >= MIN_MATCH and match_len > best_len:
                best_len, best_dist = match_len, dist

        length_code = best_len if best_len < 15 else 15
        dist_low = best_dist & 0xF if best_len > 0 else 0
        out.append((length_code << 4) | dist_low)

        if length_code == 15:
            rem = best_len - 15
            while rem >= 255:
                out.append(255); rem -= 255
            out.append(rem)

        out.extend(data[i:i + best_len])
        i += best_len

        if best_len > 0:
            if best_dist >= 15:
                rem = best_dist - 15
                while rem >= 255:
                    out.append(255); rem -= 255
                out.append(rem)
            i += best_len

    return bytes(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Cloud merging helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cloud_block_size(cloud, key):
    packed = cloud.get(key)
    if not packed:
        return None, 0, 0
    data = _lz77_decompress_auto(bytes(packed))
    count = cloud.get('count1', 0)
    if count == 0:
        return data, len(data), 0
    block = len(data) // count
    if block * count != len(data):
        return None, 0, 0
    return data, len(data), block


def _merge_cloud_fully(c1, c2):
    """Attempt full volumetric merge. Returns (merged_cloud, success)."""
    if c1.get('dim') != c2.get('dim') or \
       c1.get('bbox_origin') != c2.get('bbox_origin') or \
       c1.get('header') != c2.get('header'):
        return c1, False

    dist1, len1, bs1 = _cloud_block_size(c1, 'packed_dist')
    dist2, len2, bs2 = _cloud_block_size(c2, 'packed_dist')
    if dist1 is None or dist2 is None or bs1 != bs2 or bs1 == 0:
        return c1, False

    amb0_1, _, bs0_1 = _cloud_block_size(c1, 'packed_amb0')
    amb0_2, _, bs0_2 = _cloud_block_size(c2, 'packed_amb0')
    if amb0_1 is None or bs0_1 != bs0_2 or bs0_1 == 0:
        return c1, False

    amb1_1, _, bs1_1 = _cloud_block_size(c1, 'packed_amb1')
    amb1_2, _, bs1_2 = _cloud_block_size(c2, 'packed_amb1')
    if amb1_1 is None or bs1_1 != bs1_2 or bs1_1 == 0:
        return c1, False

    old_entries1 = c1['index_entries']
    old_entries2 = c2['index_entries']
    entry_to_idx1 = {tuple(e): i for i, e in enumerate(old_entries1)}
    entry_to_idx2 = {tuple(e): i for i, e in enumerate(old_entries2)}

    all_entries = list(dict.fromkeys(old_entries1 + old_entries2))
    new_count = len(all_entries)
    new_dist = bytearray(new_count * bs1)
    new_amb0 = bytearray(new_count * bs0_1)
    new_amb1 = bytearray(new_count * bs1_1)

    for i, entry in enumerate(all_entries):
        k = tuple(entry)
        off_d = i * bs1
        off_a0 = i * bs0_1
        if k in entry_to_idx1 and k in entry_to_idx2:
            idx1, idx2 = entry_to_idx1[k], entry_to_idx2[k]
            p1 = idx1 * bs1; p2 = idx2 * bs1
            for b in range(bs1):
                new_dist[off_d + b] = max(dist1[p1 + b], dist2[p2 + b])
            p1a = idx1 * bs0_1; p2a = idx2 * bs0_1
            for b in range(bs0_1):
                new_amb0[off_a0 + b] = (amb0_1[p1a + b] + amb0_2[p2a + b]) // 2
            p1_1 = idx1 * bs1_1; p2_1 = idx2 * bs1_1
            for b in range(bs1_1):
                new_amb1[off_d + b] = (amb1_1[p1_1 + b] + amb1_2[p2_1 + b]) // 2
        elif k in entry_to_idx1:
            idx1 = entry_to_idx1[k]
            p1 = idx1 * bs1; new_dist[off_d:off_d+bs1] = dist1[p1:p1+bs1]
            p1a = idx1 * bs0_1; new_amb0[off_a0:off_a0+bs0_1] = amb0_1[p1a:p1a+bs0_1]
            p1_1 = idx1 * bs1_1; new_amb1[off_d:off_d+bs1_1] = amb1_1[p1_1:p1_1+bs1_1]
        else:
            idx2 = entry_to_idx2[k]
            p2 = idx2 * bs1; new_dist[off_d:off_d+bs1] = dist2[p2:p2+bs1]
            p2a = idx2 * bs0_1; new_amb0[off_a0:off_a0+bs0_1] = amb0_2[p2a:p2a+bs0_1]
            p2_1 = idx2 * bs1_1; new_amb1[off_d:off_d+bs1_1] = amb1_2[p2_1:p2_1+bs1_1]

    dim = c1['dim']
    sparse_total = dim[0] * dim[1] * dim[2]
    new_sparse = []
    for x, y, z in all_entries:
        new_sparse.append([x, y, z, 0])

    new_packed_dist = list(_lz77_compress(bytes(new_dist)))
    new_packed_amb0 = list(_lz77_compress(bytes(new_amb0)))
    new_packed_amb1 = list(_lz77_compress(bytes(new_amb1)))

    merged_cloud = {
        'present': True,
        'header': c1['header'],
        'header_signed': c1['header_signed'],
        'bbox_origin': c1['bbox_origin'],
        'dim': dim,
        'sparse_non_empty': new_sparse,
        'sparse_total': sparse_total,
        'count1': new_count,
        'index_entries': all_entries,
        'v3_sizes': [len(new_packed_dist), len(new_packed_amb0), len(new_packed_amb1)],
        'packed_dist': new_packed_dist,
        'packed_amb0': new_packed_amb0,
        'packed_amb1': new_packed_amb1,
        'float_v5': c1.get('float_v5', 0.0),
        'v5_single': c1.get('v5_single', 0),
        'v6': c1.get('v6', [0, 0, 0]),
    }
    return merged_cloud, True


def _merge_cloud_v6(cloud, c2):
    """Merge v6 mesh vertices/indices from c2 into cloud (in-place)."""
    v6_1 = cloud.get('v6', [0,0,0])
    v6_2 = c2.get('v6', [0,0,0])
    v6v1 = cloud.get('v6_v')
    v6v2 = c2.get('v6_v')
    v6i1 = cloud.get('v6_indices', [])
    v6i2 = c2.get('v6_indices', [])

    if v6v2 and v6v2.get('px'):
        if not v6v1:
            cloud['v6_v'] = copy.deepcopy(v6v2)
        else:
            for attr in ['px','py','pz','nx','ny','nz','normal_u','normal_v']:
                if attr in v6v1 and attr in v6v2:
                    v6v1[attr].extend(copy.deepcopy(v6v2[attr]))
            cloud['v6_v'] = v6v1

    if v6i2:
        offset = len(cloud.get('v6_v', {}).get('px', []))
        adjusted = [i + offset for i in v6i2]
        cloud['v6_indices'] = v6i1 + adjusted

    new_v6 = list(cloud.get('v6', [0,0,0]))
    if cloud.get('v6_v'):
        new_v6[1] = len(cloud['v6_v']['px'])
    new_v6[2] = len(cloud.get('v6_indices', []))
    cloud['v6'] = new_v6


# ═══════════════════════════════════════════════════════════════════════════════
# Merge two JSONs — public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def merge_json(json1_path, json2_path, output_path):
    with open(json1_path, 'r', encoding='utf-8') as f:
        j1 = json.load(f)
    with open(json2_path, 'r', encoding='utf-8') as f:
        j2 = json.load(f)

    merged = copy.deepcopy(j1)

    for key in j2:
        if key == 'generic_meshes':
            print("  [merge] skipping generic_meshes from json2")
            continue
        if key == '_meta':
            continue

        val2 = j2[key]
        if key not in merged:
            merged[key] = copy.deepcopy(val2)
            print(f"  [merge] added new top-level key: {key}")
            continue

        val1 = merged[key]

        if key == 'terrain_patches':
            offset = len(val1)
            for idx, patch in enumerate(val2):
                new_patch = copy.deepcopy(patch)
                new_patch['index'] = offset + idx
                val1.append(new_patch)
            print(f"  [merge] terrain_patches: appended {len(val2)} patches")
            continue

        if key == 'cloud':
            c1, c2 = val1, val2
            present1 = isinstance(c1, dict) and c1.get('present', False)
            present2 = isinstance(c2, dict) and c2.get('present', False)
            if present1 and present2:
                merged_cloud, ok = _merge_cloud_fully(c1, c2)
                if ok:
                    _merge_cloud_v6(merged_cloud, c2)
                    merged['cloud'] = merged_cloud
                    print("  [merge] cloud: FULL volumetric merge succeeded")
                else:
                    print("  [merge] cloud: incompatible – keeping json1 volume, merging v6 mesh")
                    merged_cloud = copy.deepcopy(c1)
                    _merge_cloud_v6(merged_cloud, c2)
                    merged['cloud'] = merged_cloud
            elif present2:
                merged['cloud'] = copy.deepcopy(c2)
                print("  [merge] cloud: using json2 cloud (json1 absent)")
            else:
                print("  [merge] cloud: both absent")
            continue

        if key == 'cloud_mesh':
            if isinstance(val2, dict) and isinstance(val1, dict):
                groups2 = val2.get('groups', [])
                if groups2:
                    if 'groups' not in val1:
                        val1['groups'] = []
                    offset = len(val1['groups'])
                    for g in groups2:
                        new_g = copy.deepcopy(g)
                        val1['groups'].append(new_g)
                    print(f"  [merge] cloud_mesh: appended {len(groups2)} groups")
                if 'mesh_16' not in val1 and 'mesh_16' in val2:
                    val1['mesh_16'] = copy.deepcopy(val2['mesh_16'])
                    print("  [merge] cloud_mesh: added mesh_16 from json2")
            continue

        if isinstance(val1, list) and isinstance(val2, list):
            val1.extend(copy.deepcopy(val2))
            print(f"  [merge] extended list for '{key}'")
        else:
            print(f"  [merge] keeping json1 value for '{key}'")

    merged['_meta'] = merged.get('_meta', {})
    merged['_meta']['merged'] = True
    merged['_meta']['source_files'] = [
        os.path.basename(json1_path),
        os.path.basename(json2_path)
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged JSON saved to: {output_path}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════

def print_usage():
    print("""
Sky: Children of the Light — LVL/MESHES 100% Decoder/Encoder/Merger
=====================================================================
USAGE:
  python 13.py decode  input.meshes [output.json]
  python 13.py encode  input.json   [output.meshes]
  python 13.py import-obj <obj> <json> [output.json]
  python 13.py export-obj <json> [output_dir]
  python 13.py merge   <json1> <json2> <output.json>
  python 13.py help
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == 'decode':
        inp = sys.argv[2] if len(sys.argv) > 2 else None
        out = sys.argv[3] if len(sys.argv) > 3 else None
        if not inp:
            print("Usage: python 13.py decode <input.meshes> [output.json]")
            sys.exit(1)
        if not out:
            out = os.path.splitext(inp)[0] + '.json'

        with open(inp, 'rb') as f:
            magic = f.read(4)
        if magic == b'LVL0':
            tmp_dir = os.path.join(os.path.dirname(out) or '.', '_tmp_decode')
            decompressed = decompress_lvl(inp, tmp_dir)
            if decompressed:
                meshes_to_json(decompressed[0], out)
            else:
                print("Decompression failed, trying direct decode...")
                meshes_to_json(inp, out)
        else:
            meshes_to_json(inp, out)

    elif cmd == 'encode':
        inp = sys.argv[2] if len(sys.argv) > 2 else None
        out = sys.argv[3] if len(sys.argv) > 3 else None
        if not inp:
            print("Usage: python 13.py encode <input.json> [output.meshes]")
            sys.exit(1)
        if not out:
            out = os.path.splitext(inp)[0] + '.meshes'
        json_to_meshes(inp, out, as_lvl=True)

    elif cmd == 'import-obj' or cmd == 'import_obj':
        if len(sys.argv) < 4:
            print("Usage: python 13.py import-obj <obj_path> <json_path> [output.json]")
            sys.exit(1)
        obj_path = sys.argv[2]
        json_path = sys.argv[3]
        out_json = sys.argv[4] if len(sys.argv) > 4 else json_path
        import_obj_to_json(obj_path, json_path, out_json)

    elif cmd == 'export-obj' or cmd == 'export_obj':
        inp = sys.argv[2] if len(sys.argv) > 2 else None
        out_dir = sys.argv[3] if len(sys.argv) > 3 else 'obj_export'
        if not inp:
            print("Usage: python 13.py export-obj <json_path> [output_dir]")
            sys.exit(1)
        json_to_obj(inp, out_dir)

    elif cmd == 'merge':
        if len(sys.argv) < 5:
            print("Usage: python 13.py merge <json1> <json2> <output.json>")
            sys.exit(1)
        j1 = sys.argv[2]
        j2 = sys.argv[3]
        out = sys.argv[4]
        merge_json(j1, j2, out)

    elif cmd == 'help' or cmd == '-h' or cmd == '--help':
        print_usage()

    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)


# ============================================================
#  meshopt  ( Python,  meshoptimizer v1.1)
# ============================================================
K_VERTEX_HEADER = 0xA0
K_VERTEX_BLOCK_SIZE_BYTES = 8192
K_VERTEX_BLOCK_MAX_SIZE = 256
K_BYTE_GROUP_SIZE = 16

_REVERSE_BITS8 = [int(format(i, "08b")[::-1], 2) for i in range(256)]


def _get_vertex_block_size(vertex_size):
    result = (K_VERTEX_BLOCK_SIZE_BYTES // vertex_size) & ~(K_BYTE_GROUP_SIZE - 1)
    return result if result < K_VERTEX_BLOCK_MAX_SIZE else K_VERTEX_BLOCK_MAX_SIZE


def _decode_bytes_group(data, pos, out, out_off, bits):
    if bits == 0:
        for i in range(16):
            out[out_off + i] = 0
        return pos
    if bits == 8:
        out[out_off:out_off + 16] = data[pos:pos + 16]
        return pos + 16

    sentinel = (1 << bits) - 1
    byte_size = 8 // bits
    fixed_count = 16 // byte_size
    var_pos = pos + fixed_count
    idx = out_off

    for fb in range(fixed_count):
        byte = data[pos + fb]
        if bits == 1:
            byte = _REVERSE_BITS8[byte]
        for _ in range(byte_size):
            enc = byte >> (8 - bits)
            byte = (byte << bits) & 0xFF
            if enc == sentinel:
                out[idx] = data[var_pos]
                var_pos += 1
            else:
                out[idx] = enc
            idx += 1
    return var_pos


def _decode_bytes(data, pos, buffer_size, bits_table):
    num_groups = buffer_size // 16
    header_size = (num_groups + 3) // 4
    header = data[pos:pos + header_size]
    pos += header_size
    out = bytearray(buffer_size)
    for g in range(num_groups):
        bitsk = (header[g // 4] >> ((g % 4) * 2)) & 3
        pos = _decode_bytes_group(data, pos, out, g * 16, bits_table[bitsk])
    return out, pos


def _decode_deltas_u8(planes, result, base, vertex_count, vertex_size, last_vertex, k):
    for kb in range(4):
        plane = planes[kb]
        p = last_vertex[k + kb]
        off = base + kb
        for i in range(vertex_count):
            v = plane[i]
            v = (((255 if (v & 1) else 0)) ^ (v >> 1)) + p & 0xFF
            result[off] = v
            p = v
            off += vertex_size


def _decode_deltas_u16(planes, result, base, vertex_count, vertex_size, last_vertex, k):
    for kb in (0, 2):
        p = last_vertex[k + kb] | (last_vertex[k + kb + 1] << 8)
        off = base + kb
        p0 = planes[kb]
        p1 = planes[kb + 1]
        for i in range(vertex_count):
            v = p0[i] | (p1[i] << 8)
            v = (((0xFFFF if (v & 1) else 0)) ^ (v >> 1)) + p & 0xFFFF
            result[off] = v & 0xFF
            result[off + 1] = (v >> 8) & 0xFF
            p = v
            off += vertex_size


def _decode_deltas_u32_xor(planes, result, base, vertex_count, vertex_size, last_vertex, k, rot):
    p = (last_vertex[k] | (last_vertex[k + 1] << 8) |
         (last_vertex[k + 2] << 16) | (last_vertex[k + 3] << 24))
    off = base
    p0, p1, p2, p3 = planes[0], planes[1], planes[2], planes[3]
    if rot == 0:
        for i in range(vertex_count):
            v = (p0[i] | (p1[i] << 8) | (p2[i] << 16) | (p3[i] << 24)) ^ p
            result[off] = v & 0xFF
            result[off + 1] = (v >> 8) & 0xFF
            result[off + 2] = (v >> 16) & 0xFF
            result[off + 3] = (v >> 24) & 0xFF
            p = v
            off += vertex_size
    else:
        rshift = 32 - rot
        for i in range(vertex_count):
            v = p0[i] | (p1[i] << 8) | (p2[i] << 16) | (p3[i] << 24)
            v = (((v << rot) | (v >> rshift)) & 0xFFFFFFFF) ^ p
            result[off] = v & 0xFF
            result[off + 1] = (v >> 8) & 0xFF
            result[off + 2] = (v >> 16) & 0xFF
            result[off + 3] = (v >> 24) & 0xFF
            p = v
            off += vertex_size


def _decode_vertex_block(data, pos, result, vertex_offset, vertex_count,
                         vertex_size, last_vertex, channels, version):
    vertex_count_aligned = (vertex_count + 15) & ~15
    control_size = 0 if version == 0 else vertex_size // 4
    control = data[pos:pos + control_size]
    pos += control_size

    planes = [None, None, None, None]
    for k in range(0, vertex_size, 4):
        ctrl_byte = 0 if version == 0 else control[k // 4]
        for j in range(4):
            ctrl = (ctrl_byte >> (j * 2)) & 3
            if ctrl == 3:
                planes[j] = bytearray(data[pos:pos + vertex_count])
                pos += vertex_count
            elif ctrl == 2:
                planes[j] = bytearray(vertex_count)
            else:
                if version == 0:
                    bits_table = (0, 2, 4, 8)
                else:
                    bits_table = (0, 1, 2, 4) if ctrl == 0 else (1, 2, 4, 8)
                planes[j], pos = _decode_bytes(data, pos, vertex_count_aligned, bits_table)

        channel = 0 if version == 0 else channels[k // 4]
        ctype = channel & 3
        base = vertex_offset * vertex_size + k
        if ctype == 0:
            _decode_deltas_u8(planes, result, base, vertex_count, vertex_size, last_vertex, k)
        elif ctype == 1:
            _decode_deltas_u16(planes, result, base, vertex_count, vertex_size, last_vertex, k)
        else:
            rot = (32 - (channel >> 4)) & 31
            _decode_deltas_u32_xor(planes, result, base, vertex_count, vertex_size, last_vertex, k, rot)

    last_start = vertex_offset * vertex_size + (vertex_count - 1) * vertex_size
    last_vertex[:vertex_size] = result[last_start:last_start + vertex_size]
    return pos


def meshopt_decode_vertex_buffer(vertex_count, vertex_size, data):
    """ meshopt ,  (bytearray, version)"""
    if vertex_size % 4 != 0:
        raise ValueError("vertex size  4 ")

    data_end = len(data)
    if data_end < 1:
        raise ValueError("meshopt ")

    header = data[0]
    if (header & 0xF0) != K_VERTEX_HEADER:
        raise ValueError("meshopt : 0x%02X" % header)
    version = header & 0x0F
    if version > 1:
        raise ValueError(" meshopt : %d" % version)

    tail_size = vertex_size + (0 if version == 0 else vertex_size // 4)
    tail_size_min = 32 if version == 0 else 24
    tail_size_pad = max(tail_size, tail_size_min)

    if data_end < 1 + tail_size_pad:
        raise ValueError("meshopt ")

    tail_start = data_end - tail_size
    last_vertex = bytearray(256)
    last_vertex[:vertex_size] = data[tail_start:tail_start + vertex_size]
    if version != 0:
        channels = data[tail_start + vertex_size:tail_start + vertex_size + vertex_size // 4]
    else:
        channels = None

    vertex_block_size = _get_vertex_block_size(vertex_size)
    result = bytearray(vertex_count * vertex_size)
    pos = 1
    vertex_offset = 0
    while vertex_offset < vertex_count:
        block_size = min(vertex_block_size, vertex_count - vertex_offset)
        pos = _decode_vertex_block(data, pos, result, vertex_offset, block_size,
                                   vertex_size, last_vertex, channels, version)
        vertex_offset += block_size
    return result, version


# ============================================================
#  meshopt  ( Python)
#   version 0 (, bit-packed: bits=0  / bits=8 )
#   version 1 (ctrl=3 literal, channels=0)
# ============================================================
def _zigzag8_encode(d):
    """ delta (0-255) -> zigzag """
    if d < 128:
        return (d << 1) & 0xFF
    else:
        return ((d << 1) ^ 0xFF) & 0xFF


def meshopt_encode_vertex_buffer(vertex_count, vertex_size, vertex_data, version=0):
    """
     meshopt 
    version=0:  (bit-packed, bits=0/8)
    version=1: ctrl=3 literal, channels=0
    vertex_data: bytes-like,  = vertex_count * vertex_size
    """
    if vertex_size % 4 != 0:
        raise ValueError("vertex size  4 ")
    if len(vertex_data) < vertex_count * vertex_size:
        raise ValueError("vertex_data ")

    if version == 0:
        return _encode_v0(vertex_count, vertex_size, vertex_data)
    else:
        return _encode_v1(vertex_count, vertex_size, vertex_data)


def _encode_v0(vertex_count, vertex_size, vertex_data):
    """
    Version 0  ()
    - ,  channel 0 (u8 zigzag delta)
    -  16  bit-packed ,  bits: 0/2/4/8
    - bits_table = (0, 2, 4, 8): selector 00, 12, 24, 38
    - tail:  first_vertex (vertex_size ,  channels)
    """
    out = bytearray()
    out.append(K_VERTEX_HEADER | 0)  # 0xA0

    if vertex_count == 0:
        tail = bytearray(vertex_data[:vertex_size]) if len(vertex_data) >= vertex_size else bytearray(vertex_size)
        out.extend(tail)
        return bytes(out)

    vertex_block_size = _get_vertex_block_size(vertex_size)
    tail = bytearray(vertex_data[:vertex_size])
    last_vertex = bytearray(vertex_data[:vertex_size])

    vertex_offset = 0
    while vertex_offset < vertex_count:
        block_size = min(vertex_block_size, vertex_count - vertex_offset)
        vertex_count_aligned = (block_size + 15) & ~15
        num_groups = vertex_count_aligned // 16
        header_size = (num_groups + 3) // 4

        for k in range(0, vertex_size, 4):
            for kb in range(4):
                # zigzag delta for this channel
                deltas = bytearray(block_size)
                p = last_vertex[k + kb]
                for i in range(block_size):
                    v = vertex_data[(vertex_offset + i) * vertex_size + k + kb]
                    d = (v - p) & 0xFF
                    deltas[i] = _zigzag8_encode(d)
                    p = v
                last_vertex[k + kb] = vertex_data[(vertex_offset + block_size - 1) * vertex_size + k + kb]

                header = bytearray(header_size)
                data = bytearray()

                for g in range(num_groups):
                    start = g * 16
                    end = min(start + 16, block_size)
                    # pad group to 16 with zeros
                    vals = [deltas[start + i] if start + i < block_size else 0 for i in range(16)]
                    sel, enc = _encode_group_v0(vals)
                    header[g // 4] |= (sel << ((g % 4) * 2))
                    data.extend(enc)

                out.extend(header)
                out.extend(data)

        vertex_offset += block_size

    out.extend(tail)
    return bytes(out)


def _encode_group_v0(vals):
    """Encode a 16-value group for v0. Returns (selector, encoded_bytes).
    bits_table = (0, 2, 4, 8): sel 0bits=0, sel 1bits=2, sel 2bits=4, sel 3bits=8
    """
    # bits=0: all zero
    if all(v == 0 for v in vals):
        return 0, b''

    # Compute sizes for each option
    # bits=2: sentinel=3, 4 fixed bytes + 1 var byte per value>=3
    var2 = sum(1 for v in vals if v >= 3)
    size2 = 4 + var2

    # bits=4: sentinel=15, 8 fixed bytes + 1 var byte per value>=15
    var4 = sum(1 for v in vals if v >= 15)
    size4 = 8 + var4

    # bits=8: 16 literal bytes
    size8 = 16

    # Pick smallest
    if size2 <= size4 and size2 <= size8:
        return 1, _pack_bits(vals, 2)
    if size4 <= size8:
        return 2, _pack_bits(vals, 4)
    return 3, bytes(vals)


def _pack_bits(vals, bits):
    """Pack 16 values into bit-packed format (MSB-first, sentinel for overflow).
    Returns fixed_bytes + variable_bytes.
    """
    sentinel = (1 << bits) - 1
    byte_size = 8 // bits       # values per fixed byte
    fixed_count = 16 // byte_size
    fixed = bytearray(fixed_count)
    variable = bytearray()

    for i in range(16):
        v = vals[i]
        byte_idx = i // byte_size
        pos_in_byte = i % byte_size
        shift = 8 - bits * (pos_in_byte + 1)
        if v >= sentinel:
            fixed[byte_idx] |= (sentinel << shift)
            variable.append(v)
        else:
            fixed[byte_idx] |= (v << shift)

    return bytes(fixed) + bytes(variable)


def _encode_v1(vertex_count, vertex_size, vertex_data):
    """Version 1 : ctrl=3 literal, channels=0 ()"""
    if vertex_count == 0:
        tail = bytearray(vertex_data[:vertex_size]) + bytearray(vertex_size // 4)
        return bytes([K_VERTEX_HEADER | 1]) + bytes(tail)

    out = bytearray()
    out.append(K_VERTEX_HEADER | 1)  # 0xA1

    vertex_block_size = _get_vertex_block_size(vertex_size)
    control_size = vertex_size // 4
    tail = bytearray(vertex_data[:vertex_size]) + bytearray(control_size)
    last_vertex = bytearray(vertex_data[:vertex_size])

    vertex_offset = 0
    while vertex_offset < vertex_count:
        block_size = min(vertex_block_size, vertex_count - vertex_offset)
        out.extend(b'\xff' * control_size)

        for k in range(0, vertex_size, 4):
            for kb in range(4):
                p = last_vertex[k + kb]
                for i in range(block_size):
                    v = vertex_data[(vertex_offset + i) * vertex_size + k + kb]
                    d = (v - p) & 0xFF
                    out.append(_zigzag8_encode(d))
                    p = v
                last_vertex[k + kb] = vertex_data[(vertex_offset + block_size - 1) * vertex_size + k + kb]

        vertex_offset += block_size

    out.extend(tail)
    return bytes(out)



# ================================================================================
# UNIVERSAL CONTAINER + VERSION-GATED CODECS (v2)
# --------------------------------------------------------------------------------
# Adds support for every known Sky: Children of the Light .meshes container:
#   0x33 .. 0x3D  (51 .. 61)
# The definitions below override the legacy single-version implementations that
# used to live earlier in this file.
# ================================================================================

import base64
import hashlib
import array
import struct

if 'meshopt_decode_vertex_buffer' not in globals():
    from _mopt_clean import meshopt_decode_vertex_buffer, meshopt_encode_vertex_buffer

# ---- version constants ---------------------------------------------------------
V_LEX = 0x33           # 51 : empty / prototype levels (WorldEmpty)
V_FIRST_LOD = 0x34     # 52 : LOD0 only
V_METR = 0x36          # 54 : LOD0 + METR (and a trailing cloud bool)
V_CLOUD_PROXY = 0x37   # 55 : cloud has meshopt proxy + meshopt skirts
V_GEO0 = 0x38          # 56 : first GEO0 section (opaque format), no LOD terrain
V_GEO0_FULL = 0x39     # 57 : GEO0 with full patch/chunk tables
V_CLOUD_NOMASK = 0x3C  # 60 : cloud has no bitmask and four blobs
SUPPORTED_VERSION_MIN = 0x33
SUPPORTED_VERSION_MAX = 0x3D

# Where each section stops being compressed / how it is stored
RAW_SECTIONS = ('GEO0', 'METR')

SECTIONS_MODERN = ('GEO0', 'LOD0', 'METR')
SECTIONS_METR = ('LOD0', 'METR')


class MeshesFormatError(Exception):
    """Raised when a .meshes buffer does not match any known layout."""


# ---- small helpers -------------------------------------------------------------
def _b64e(data) -> str:
    return base64.b64encode(bytes(data)).decode('ascii')


def _b64d(text) -> bytes:
    if not text:
        return b''
    return base64.b64decode(text.encode('ascii'))


def _md5(data) -> str:
    return hashlib.md5(bytes(data)).hexdigest()


def _u32(data, off) -> int:
    return struct.unpack_from('<I', data, off)[0]


def _f32(data, off) -> float:
    return struct.unpack_from('<f', data, off)[0]


def _version_name(version: int) -> str:
    names = {
        0x33: '0.3.x prototype',
        0x34: '0.4.0 - 0.4.1',
        0x35: '0.4.2 - 0.4.6',
        0x36: '0.4.7 - 0.15.5',
        0x37: '0.16.0 - 0.17.6',
        0x38: '0.18.0 - 0.18.1',
        0x39: '0.18.6 - 0.19.0',
        0x3A: '0.19.5',
        0x3B: '0.19.5 - 0.23.0',
        0x3C: '0.23.4 - 0.34.3',
        0x3D: '0.33.5 - 0.35.2',
    }
    return names.get(version, 'unknown')


def _lz4_decompress_any(data: bytes) -> bytes:
    """Decompress an LZ4 *block* whose original size is not stored.
    The legacy code used a hardcoded 12 MiB buffer; modern LOD0 payloads can be
    larger, so grow the buffer until liblz4 accepts it."""
    size = 1 << 20
    last = None
    while size <= (1 << 27):
        try:
            return lz4.block.decompress(data, uncompressed_size=size)
        except Exception as exc:  # noqa: BLE001 - liblz4 reports size mismatch
            last = exc
            size <<= 1
    raise MeshesFormatError('LZ4 decompression failed: %s' % last)


def _lz4_compress(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


# ---- container -----------------------------------------------------------------
def parse_container(data: bytes) -> dict:
    """Parse the LVL0 wrapper that every .meshes file uses.

    Layout (verified on 0x33 .. 0x3D):
      0x00  'LVL0'
      0x04  u32 version
      0x08  u8  entry_count
      0x09  3 padding bytes (ff ff ff when old, 00 00 00 when new)
      0x0C  entry_count * 12 bytes: 4cc name, u32 offset, u32 length
      ...   padding / uninitialised bytes
      first-24: 6 * f32 bounds (min xyz, max xyz)
      first:    first section payload
    """
    if len(data) < 12 or data[0:4] != b'LVL0':
        raise MeshesFormatError('not a LVL0 container')
    version = _u32(data, 4)
    count = data[8]
    if count == 0 or count > 16:
        raise MeshesFormatError('unexpected TOC entry count: %d' % count)
    entries = []
    for i in range(count):
        base = 0x0C + i * 12
        if base + 12 > len(data):
            raise MeshesFormatError('TOC truncated at entry %d' % i)
        name = bytes(data[base:base + 4])
        offset = _u32(data, base + 4)
        length = _u32(data, base + 8)
        if offset < 12 or offset + length > len(data):
            raise MeshesFormatError('TOC entry %r out of bounds' % name)
        entries.append({'name': name, 'offset': offset, 'length': length,
                        'name_str': name.decode('ascii', 'replace').rstrip('\x00')})
    toc_end = 0x0C + count * 12
    first = min(e['offset'] for e in entries)
    if first < 24:
        raise MeshesFormatError('first section offset too small: 0x%x' % first)
    gap = bytes(data[toc_end:first - 24]) if first - 24 > toc_end else b''
    bounds = list(struct.unpack_from('<6f', data, first - 24))
    return {
        'version': version,
        'count': count,
        'pad': bytes(data[9:12]),
        'entries': entries,
        'toc_end': toc_end,
        'gap': gap,
        'bounds': bounds,
        'first': first,
    }


def read_section_payloads(data: bytes, container: dict) -> dict:
    out = {}
    for entry in container['entries']:
        out[entry['name_str']] = bytes(data[entry['offset']:entry['offset'] + entry['length']])
    return out


def build_container(version: int, sections: list, bounds, pad=None, gap=None) -> bytes:
    """Build a LVL0 wrapper.

    sections : list of (4-char name, payload bytes) in TOC order.
    bounds   : 6 floats or None (zeros).
    pad/gap  : optional exact bytes captured from the source file, so an
               untouched decode->encode roundtrip is byte identical.
    """
    count = len(sections)
    toc_end = 0x0C + count * 12
    if gap is None:
        canonical_first = 0x84 if (count == 1 and version <= 0x34) else 0x88
        gap_len = max(0, canonical_first - 24 - toc_end)
        gap = b'\x00' * gap_len
    first = toc_end + len(gap) + 24
    if pad is None:
        pad = b'\xff\xff\xff' if version <= 0x36 else b'\x00\x00\x00'
    if len(pad) != 3:
        pad = (bytes(pad) + b'\x00\x00\x00')[:3]

    header = bytearray()
    header += b'LVL0'
    header += struct.pack('<I', version & 0xFFFFFFFF)
    header.append(count & 0xFF)
    header += pad
    offset = first
    for name, payload in sections:
        name4 = (name.encode('ascii', 'replace') + b'\x00\x00\x00\x00')[:4]
        header += name4
        header += struct.pack('<II', offset, len(payload))
        offset += len(payload)

    out = bytearray(header)
    out += gap
    if bounds is None:
        bounds = [0.0] * 6
    out += struct.pack('<6f', *bounds)
    for _name, payload in sections:
        out += payload
    return bytes(out)


def default_section_order(decoded: dict) -> list:
    order = []
    if decoded.get('geo0'):
        order.append('GEO0')
    order.append('LOD0')
    if decoded.get('metr'):
        order.append('METR')
    return order


# ---- LOD0 ----------------------------------------------------------------------
def decode_lod0(raw: bytes, version: int) -> dict:
    """Decode the LZ4 payload of the LOD0 section for `version`."""
    stream = BinaryStreamReader(raw)
    out = {}

    out['generic_meshes'] = decode_generic_meshes(stream)
    if version <= V_CLOUD_PROXY:
        # 0x33 .. 0x37 store the terrain inside the LOD payload
        out['terrain_patches'] = decode_terrain_patches(stream)
    else:
        # 0x38+ move terrain into the (raw) GEO0 section
        out['terrain_patches'] = []
    out['cloud'] = decode_cloud_section(stream, version)
    out['cloud_mesh'] = decode_cloud_mesh(stream, version)

    remaining = stream.remaining()
    if remaining:
        out['_remaining'] = list(stream.read_bytes(remaining))
    return out


def encode_lod0(decoded: dict, version: int) -> bytes:
    writer = BinaryStreamWriter()
    encode_generic_meshes(writer, decoded.get('generic_meshes', []))
    if version <= V_CLOUD_PROXY:
        encode_terrain_patches(writer, decoded.get('terrain_patches', []))
    encode_cloud_section(writer, decoded.get('cloud', {'present': False}), version)
    encode_cloud_mesh(writer, decoded.get('cloud_mesh', {}), version)
    if decoded.get('_remaining'):
        writer.write_bytes(bytes(decoded['_remaining']))
    return writer.to_bytes()


# ---- cloud volume --------------------------------------------------------------
def _decode_coords(stream, count):
    raw = stream.read_bytes(count * 6)
    return [list(struct.unpack_from('<3h', raw, i * 6)) for i in range(count)]


def _encode_coords(writer, coords):
    for x, y, z in coords:
        writer.write_bytes(struct.pack('<3h', int(x), int(y), int(z)))


def decode_cloud_section(stream: BinaryStreamReader, version: int = 0x36) -> dict:
    """Version gated volumetric cloud block."""
    stream.align()
    flag = stream.read_uint32()
    if flag in (0, 0xFFFFFFFF):
        return {'present': False, 'version': version}

    result = {'present': True, 'version': version}
    header = [stream.read_uint32() for _ in range(6)]
    signed = [x - 0x100000000 if x >= 0x80000000 else x for x in header]
    result['header'] = header
    result['header_signed'] = signed
    result['bbox_origin'] = signed[0:3]
    dim = header[3:6]
    result['dim'] = dim
    gx, gy, gz = dim
    if abs(gx * gy * gz) > 64_000_000:
        raise MeshesFormatError('invalid cloud grid dimensions %r' % (dim,))

    if version <= 0x36:
        _decode_cloud_legacy(stream, result)
    else:
        _decode_cloud_modern(stream, result, version)
    return result


def _decode_cloud_legacy(stream: BinaryStreamReader, result: dict):
    """0x33 .. 0x36: one mask byte per cell, 16 byte proxy vertices."""
    gx, gy, gz = result['dim']
    cells = abs(gx * gy * gz)
    raw = stream.read_bytes(cells)
    non_empty = []
    for idx in range(cells):
        value = raw[idx]
        if value != 0xFF:
            z = idx // (gx * gy)
            y = (idx // gx) % gy
            x = idx % gx
            non_empty.append([x, y, z, value])
    result['sparse_non_empty'] = non_empty
    result['sparse_total'] = cells

    count1 = stream.read_uint32()
    result['count1'] = count1
    result['index_entries'] = _decode_coords(stream, count1)

    sizes = [stream.read_uint32() for _ in range(3)]
    result['v3_sizes'] = sizes
    for name, size in zip(('packed_dist', 'packed_amb0', 'packed_amb1'), sizes):
        result[name] = list(stream.read_bytes(size))

    result['voxel_scale'] = stream.read_float()
    result['bpp_dist_amb01'] = stream.read_uint32()

    v6 = [stream.read_uint32() for _ in range(3)]
    result['v6'] = v6
    vert_count = v6[1]
    index_count = v6[2]
    raw = stream.read_bytes(16 * vert_count)
    px, py, pz = [], [], []
    nu, nv = [], []
    for i in range(vert_count):
        offset = i * 16
        x, y, z = struct.unpack_from('<3f', raw, offset)
        packed = _u32(raw, offset + 12)
        px.append(x)
        py.append(y)
        pz.append(z)
        nu.append(packed & 0xFFFF)
        nv.append((packed >> 16) & 0xFFFF)
    result['v6_v'] = {'px': px, 'py': py, 'pz': pz, 'normal_u': nu, 'normal_v': nv}
    result['v6_indices'] = list(
        struct.unpack('<%dH' % index_count, stream.read_bytes(2 * index_count)))
    if result['version'] >= 0x36:
        result['trail'] = stream.read_bytes(1)[0]


def _decode_cloud_modern(stream: BinaryStreamReader, result: dict, version: int):
    """0x37+: two mask bytes per cell (except 0x3C+), meshopt proxy for 0x37."""
    gx, gy, gz = result['dim']
    cells = abs(gx * gy * gz)
    if version < V_CLOUD_NOMASK:
        raw = stream.read_bytes(cells * 2)
        result['mask'] = list(struct.unpack('<%dH' % cells, raw))
        result['mask_total'] = cells

    count1 = stream.read_uint32()
    result['count1'] = count1
    result['index_entries'] = _decode_coords(stream, count1)

    blob_names = ['packed_dist', 'packed_amb0', 'packed_amb1']
    if version >= V_CLOUD_NOMASK:
        blob_names.append('packed_extra')
    sizes = [stream.read_uint32() for _ in blob_names]
    result['v3_sizes'] = sizes
    for name, size in zip(blob_names, sizes):
        result[name] = list(stream.read_bytes(size))

    result['voxel_scale'] = stream.read_float()
    result['bpp_dist_amb01'] = stream.read_uint32()
    result['bpp_amb2'] = stream.read_uint32()

    if version == V_CLOUD_PROXY:
        a, vert_count, index_count, meta_size, vb_size, ib_size = (
            stream.read_uint32() for _ in range(6))
        proxy = {'a': a, 'vert_count': vert_count, 'index_count': index_count,
                 'meta_size': meta_size, 'vb_size': vb_size, 'ib_size': ib_size}
        result['proxy'] = proxy
        proxy['meta'] = list(stream.read_bytes(meta_size))
        _decode_meshopt_vb(stream, proxy, vert_count, 20, 'proxy')
        proxy['ib_b64'] = _b64e(stream.read_bytes(ib_size))
        result['trail'] = stream.read_bytes(1)[0]


def encode_cloud_section(writer: BinaryStreamWriter, cloud: dict, version: int = 0x36):
    if not cloud or not cloud.get('present'):
        writer.write_uint32(0)
        return
    writer.write_uint32(1)
    signed = cloud.get('bbox_origin', [0, 0, 0])
    dim = cloud.get('dim', [0, 0, 0])
    for value in list(signed)[:3]:
        writer.write_uint32(value & 0xFFFFFFFF)
    for value in list(dim)[:3]:
        writer.write_uint32(value & 0xFFFFFFFF)

    if version <= 0x36:
        _encode_cloud_legacy(writer, cloud, version)
    else:
        _encode_cloud_modern(writer, cloud, version)


def _encode_cloud_legacy(writer: BinaryStreamWriter, cloud: dict, version: int):
    gx, gy, gz = cloud.get('dim', [0, 0, 0])
    cells = abs(gx * gy * gz)
    sparse_total = cloud.get('sparse_total', cells)
    sparse = bytearray([0xFF]) * sparse_total
    for x, y, z, value in cloud.get('sparse_non_empty', []):
        idx = z * gx * gy + y * gx + x
        if 0 <= idx < len(sparse):
            sparse[idx] = value & 0xFF
    writer.write_bytes(bytes(sparse))

    writer.write_uint32(cloud.get('count1', 0))
    _encode_coords(writer, cloud.get('index_entries', []))

    sizes = list(cloud.get('v3_sizes') or [0, 0, 0])
    for value in sizes[:3]:
        writer.write_uint32(value)
    for name in ('packed_dist', 'packed_amb0', 'packed_amb1'):
        writer.write_bytes(bytes(cloud.get(name, [])))

    writer.write_float(cloud.get('voxel_scale', 0.0))
    writer.write_uint32(cloud.get('bpp_dist_amb01', 4))

    v6 = list(cloud.get('v6') or [0, 0, 0])
    for value in v6[:3]:
        writer.write_uint32(value)
    v6v = cloud.get('v6_v') or {}
    for i in range(v6[1] if len(v6) > 1 else 0):
        writer.write_bytes(struct.pack('<3f', v6v['px'][i], v6v['py'][i], v6v['pz'][i]))
        writer.write_uint32(v6v['normal_u'][i] | (v6v['normal_v'][i] << 16))
    for value in cloud.get('v6_indices', []):
        writer.write_bytes(struct.pack('<H', value))

    if version >= 0x36:
        writer.write_bytes(bytes([cloud.get('trail', 0) & 0xFF]))


def _encode_cloud_modern(writer: BinaryStreamWriter, cloud: dict, version: int):
    gx, gy, gz = cloud.get('dim', [0, 0, 0])
    cells = abs(gx * gy * gz)
    if version < V_CLOUD_NOMASK:
        mask = cloud.get('mask') or []
        if len(mask) < cells:
            mask = list(mask) + [0] * (cells - len(mask))
        writer.write_bytes(array.array('H', [int(v) & 0xFFFF for v in mask[:cells]]).tobytes())

    writer.write_uint32(cloud.get('count1', 0))
    _encode_coords(writer, cloud.get('index_entries', []))

    blob_names = ['packed_dist', 'packed_amb0', 'packed_amb1']
    if version >= V_CLOUD_NOMASK:
        blob_names.append('packed_extra')
    sizes = list(cloud.get('v3_sizes') or [])
    if len(sizes) < len(blob_names):
        sizes += [len(cloud.get(name, [])) for name in blob_names[len(sizes):]]
    for value in sizes[:len(blob_names)]:
        writer.write_uint32(value)
    for name in blob_names:
        writer.write_bytes(bytes(cloud.get(name, [])))

    writer.write_float(cloud.get('voxel_scale', 0.0))
    writer.write_uint32(cloud.get('bpp_dist_amb01', 4))
    writer.write_uint32(cloud.get('bpp_amb2', 2))

    if version == V_CLOUD_PROXY:
        proxy = cloud.get('proxy') or {}
        meta = bytes(proxy.get('meta', []))
        vb = render_meshopt_vb(proxy, 20)
        ib = _b64d(proxy.get('ib_b64', ''))
        writer.write_uint32(proxy.get('a', 0))
        writer.write_uint32(proxy.get('vert_count', 0))
        writer.write_uint32(proxy.get('index_count', 0))
        writer.write_uint32(len(meta))
        writer.write_uint32(len(vb))
        writer.write_uint32(len(ib))
        writer.write_bytes(meta)
        writer.write_bytes(vb)
        writer.write_bytes(ib)
        writer.write_bytes(bytes([cloud.get('trail', 0) & 0xFF]))



def _decode_meshopt_vb(stream: BinaryStreamReader, target: dict, count: int, stride: int, label: str):
    """Read + meshopt decode a vertex buffer, storing editable positions."""
    size = target.get('vb_size', 0)
    data = stream.read_bytes(size)
    target['vb_comp_b64'] = _b64e(data)
    if count <= 0 or size <= 0:
        target['v'] = {'px': [], 'py': [], 'pz': [], 'rest_b64': ''}
        target['stride'] = stride
        target['vb_hash'] = ''
        return
    try:
        raw, _ver = meshopt_decode_vertex_buffer(count, stride, data)
    except Exception as exc:  # noqa: BLE001 - keep the blob verbatim
        target['vb_error'] = str(exc)
        target['v'] = {'px': [], 'py': [], 'pz': [], 'rest_b64': ''}
        target['stride'] = stride
        target['vb_hash'] = ''
        return
    raw = bytes(raw)
    target['stride'] = stride
    target['vb_hash'] = _md5(raw)
    target['v'] = _split_verts(raw, count, stride)


def render_meshopt_vb(target: dict, stride: int) -> bytes:
    """Return the meshopt payload for a vertex buffer.  Unchanged bytes are
    reused verbatim so a decode->encode roundtrip is bit exact; edited vertices
    are re-encoded with the embedded pure python meshopt encoder."""
    v = target.get('v') or {}
    count = len(v.get('px', []))
    stored = _b64d(target.get('vb_comp_b64', ''))
    raw = _join_verts(v, stride)
    if raw is not None and target.get('vb_hash') and _md5(raw) == target['vb_hash'] and stored:
        return stored
    if raw is not None and count > 0:
        return bytes(meshopt_encode_vertex_buffer(count, stride, raw))
    return stored


def _encode_meshopt_vb(writer: BinaryStreamWriter, target: dict, stride: int) -> bytes:
    out = render_meshopt_vb(target, stride)
    writer.write_uint32(len(out))
    writer.write_bytes(out)
    return out


# ---- skirts + occluder ---------------------------------------------------------
def _triangles_from_indices(indices):
    return [[indices[i], indices[i + 1], indices[i + 2]]
            for i in range(0, len(indices) - 2, 3)]


def decode_cloud_mesh(stream: BinaryStreamReader, version: int = 0x36) -> dict:
    result = {'version': version, 'groups': [], 'mesh_16': None}
    count = stream.read_uint32()
    if count > 1_000_000:
        raise MeshesFormatError('implausible skirt count %d' % count)

    for _ in range(count):
        if version <= 0x36:
            group = _decode_raw_skirt(stream)
        else:
            group = _decode_meshopt_skirt(stream, version)
        result['groups'].append(group)

    occluder_count = stream.read_uint32()
    if occluder_count:
        vert_count = stream.read_uint32()
        index_count = stream.read_uint32()
        raw = stream.read_bytes(16 * vert_count)
        indices = list(struct.unpack('<%dH' % index_count, stream.read_bytes(2 * index_count)))
        px, py, pz, nu, nv = [], [], [], [], []
        for i in range(vert_count):
            x, y, z = struct.unpack_from('<3f', raw, i * 16)
            packed = _u32(raw, i * 16 + 12)
            px.append(x)
            py.append(y)
            pz.append(z)
            nu.append(packed & 0xFFFF)
            nv.append((packed >> 16) & 0xFFFF)
        result['mesh_16'] = {
            'vert_count': vert_count,
            'v': {'px': px, 'py': py, 'pz': pz, 'normal_u': nu, 'normal_v': nv},
            'idx_count': index_count,
            'indices': indices,
            'triangles': _triangles_from_indices(indices),
        }
    return result


def _decode_raw_skirt(stream: BinaryStreamReader) -> dict:
    vert_count = stream.read_uint32()
    raw = stream.read_bytes(40 * vert_count)
    px, py, pz = [], [], []
    nu, nv = [], []
    mat, attr = [], []
    cr, cg, cb, ca = [], [], [], []
    exa, exb, exc = [], [], []
    for i in range(vert_count):
        off = i * 40
        x, y, z = struct.unpack_from('<3f', raw, off)
        px.append(x)
        py.append(y)
        pz.append(z)
        packed = _u32(raw, off + 12)
        nu.append(packed & 0xFFFF)
        nv.append((packed >> 16) & 0xFFFF)
        mat.append(_u32(raw, off + 16))
        attr.append(_u32(raw, off + 20))
        cr.append(raw[off + 24])
        cg.append(raw[off + 25])
        cb.append(raw[off + 26])
        ca.append(raw[off + 27])
        exa.append(_u32(raw, off + 28))
        exb.append(_u32(raw, off + 32))
        exc.append(_u32(raw, off + 36))
    index_count = stream.read_uint32()
    indices = list(struct.unpack('<%dH' % index_count, stream.read_bytes(2 * index_count)))
    return {
        'vert_count': vert_count,
        'triangles': _triangles_from_indices(indices),
        'v': {'px': px, 'py': py, 'pz': pz,
              'normal_u': nu, 'normal_v': nv,
              'material_id': mat, 'vertex_attr': attr,
              'cr': cr, 'cg': cg, 'cb': cb, 'ca': ca,
              'extra_a': exa, 'extra_b': exb, 'extra_c': exc},
        'idx_count': index_count,
        'indices': indices,
    }


def _decode_meshopt_skirt(stream: BinaryStreamReader, version: int) -> dict:
    stride = 40 if version <= 0x38 else 36
    vert_count = stream.read_uint32()
    vb_size = stream.read_uint32()
    group = {'vert_count': vert_count, 'stride': stride, 'vb_size': vb_size}
    _decode_meshopt_vb(stream, group, vert_count, stride, 'skirt')
    index_count = stream.read_uint32()
    ib_size = stream.read_uint32()
    ib = stream.read_bytes(ib_size)
    group['idx_count'] = index_count
    group['ib_size'] = ib_size
    group['ib_b64'] = _b64e(ib)   # meshopt index stream, kept verbatim
    return group


def encode_cloud_mesh(writer: BinaryStreamWriter, cloud_mesh: dict, version: int = 0x36):
    groups = cloud_mesh.get('groups', []) or []
    mesh_16 = cloud_mesh.get('mesh_16')
    writer.write_uint32(len(groups))
    for group in groups:
        if version <= 0x36:
            _encode_raw_skirt(writer, group)
        else:
            writer.write_uint32(group.get('vert_count', 0))
            _encode_meshopt_vb(writer, group, 40 if version <= 0x38 else 36)
            ib = _b64d(group.get('ib_b64', ''))
            writer.write_uint32(group.get('idx_count', 0))
            writer.write_uint32(group.get('ib_size', len(ib)))
            writer.write_bytes(ib)

    if not mesh_16:
        writer.write_uint32(0)
        return
    writer.write_uint32(1)
    writer.write_uint32(mesh_16.get('vert_count', 0))
    writer.write_uint32(mesh_16.get('idx_count', 0))
    v = mesh_16.get('v', {})
    for i in range(mesh_16.get('vert_count', 0)):
        writer.write_bytes(struct.pack('<3f', v['px'][i], v['py'][i], v['pz'][i]))
        writer.write_uint32(v['normal_u'][i] | (v['normal_v'][i] << 16))
    for value in mesh_16.get('indices', []):
        writer.write_bytes(struct.pack('<H', value))


def _encode_raw_skirt(writer: BinaryStreamWriter, group: dict):
    v = group['v']
    vert_count = group.get('vert_count', 0)
    writer.write_uint32(vert_count)
    for i in range(vert_count):
        writer.write_bytes(struct.pack('<3f', v['px'][i], v['py'][i], v['pz'][i]))
        writer.write_uint32(v['normal_u'][i] | (v['normal_v'][i] << 16))
        writer.write_uint32(v['material_id'][i])
        writer.write_uint32(v['vertex_attr'][i])
        writer.write_bytes(bytes([v['cr'][i], v['cg'][i], v['cb'][i], v['ca'][i]]))
        writer.write_uint32(v['extra_a'][i])
        writer.write_uint32(v['extra_b'][i])
        writer.write_uint32(v['extra_c'][i])
    writer.write_uint32(group.get('idx_count', 0))
    for value in group.get('indices', []):
        writer.write_bytes(struct.pack('<H', value))


# ---- GEO0 ----------------------------------------------------------------------
def decode_geo0(data: bytes, version: int) -> dict:
    """Decode the raw GEO0 section.

    0x39+ : fully decoded (meshopt vertices + raw u8 index triplets + patch and
            sub chunk tables).
    0x38  : different, undocumented layout -> kept verbatim so the file can be
            re-encoded exactly, with a note in the JSON.
    """
    if not data:
        return None
    if version <= V_GEO0:
        return {
            'opaque': True,
            'note': 'GEO0 layout for 0x38 is not decoded; blob preserved verbatim',
            'raw_b64': _b64e(data),
            'header': list(struct.unpack_from('<6I', data, 0)) if len(data) >= 24 else [],
        }

    if len(data) < 24:
        return {'opaque': True, 'note': 'GEO0 shorter than header',
                'raw_b64': _b64e(data), 'header': []}

    header = list(struct.unpack_from('<6I', data, 0))
    index_bytes, vert_count, chunk_count, cloud_chunk_count, subchunk_count, vb_size = header
    expected = 24 + vb_size + index_bytes + (chunk_count + cloud_chunk_count) * 56 + subchunk_count * 8
    if expected > len(data):
        raise MeshesFormatError('GEO0 header does not fit section (%d > %d)' % (expected, len(data)))

    result = {
        'header': header,
        'vert_count': vert_count,
        'index_bytes': index_bytes,
        'chunk_count': chunk_count,
        'cloud_chunk_count': cloud_chunk_count,
        'subchunk_count': subchunk_count,
        'vb_comp_b64': _b64e(data[24:24 + vb_size]),
    }
    try:
        raw, _ver = meshopt_decode_vertex_buffer(vert_count, 36, _b64d(result['vb_comp_b64']))
        raw = bytes(raw)
        if len(raw) != vert_count * 36:
            raise MeshesFormatError('decoded vertex buffer size mismatch')
        result['vb_hash'] = _md5(raw)
        result['v'] = _split_verts(raw, vert_count, 36)
    except Exception as exc:  # noqa: BLE001
        result['vb_error'] = str(exc)
        result['vb_hash'] = ''
        result['v'] = {'px': [], 'py': [], 'pz': [], 'rest_b64': ''}

    ib_off = 24 + vb_size
    result['ib_b64'] = _b64e(data[ib_off:ib_off + index_bytes])
    table_off = ib_off + index_bytes
    chunks = []
    for i in range(chunk_count):
        chunks.append(_unpack_chunk(data, table_off + i * 56))
    result['chunks'] = chunks
    table_off += chunk_count * 56
    cloud_chunks = []
    for i in range(cloud_chunk_count):
        cloud_chunks.append(_unpack_chunk(data, table_off + i * 56))
    result['cloud_chunks'] = cloud_chunks
    table_off += cloud_chunk_count * 56
    subchunks = []
    for i in range(subchunk_count):
        offset = table_off + i * 8
        subchunks.append({
            'material_id': data[offset], 'triangle_count': data[offset + 1],
            'vtx_count': data[offset + 2], 'triangle_start': data[offset + 3],
            'triangle_end': data[offset + 4], 'vtx_start': data[offset + 5],
            'vtx_end': data[offset + 6], 'unk': data[offset + 7],
        })
    result['subchunks'] = subchunks
    table_off += subchunk_count * 8
    if table_off < len(data):
        result['trailing_b64'] = _b64e(data[table_off:])
    return result


def _unpack_chunk(data: bytes, offset: int) -> dict:
    idx_start, vtx_start, area_start = struct.unpack_from('<3I', data, offset)
    idx_count = struct.unpack_from('<H', data, offset + 12)[0]
    vtx_count = data[offset + 14]
    area_count = data[offset + 15]
    mn = list(struct.unpack_from('<3f', data, offset + 16))
    mx = list(struct.unpack_from('<3f', data, offset + 28))
    unk = [struct.unpack_from('<I', data, offset + 40 + 4 * i)[0] for i in range(4)]
    return {'idx_start': idx_start, 'vtx_start': vtx_start, 'area_start': area_start,
            'idx_count': idx_count, 'vtx_count': vtx_count, 'area_count': area_count,
            'min': mn, 'max': mx, 'unk': unk}


def _pack_chunk(chunk: dict) -> bytes:
    out = bytearray()
    out += struct.pack('<3I', int(chunk.get('idx_start', 0)), int(chunk.get('vtx_start', 0)),
                       int(chunk.get('area_start', 0)))
    out += struct.pack('<H', int(chunk.get('idx_count', 0)) & 0xFFFF)
    out.append(int(chunk.get('vtx_count', 0)) & 0xFF)
    out.append(int(chunk.get('area_count', 0)) & 0xFF)
    out += struct.pack('<6f', *list(chunk.get('min', [0, 0, 0]))[:3] +
                       list(chunk.get('max', [0, 0, 0]))[:3])
    unk = list(chunk.get('unk', [0, 0, 0, 0]))[:4]
    while len(unk) < 4:
        unk.append(0)
    out += struct.pack('<4I', *[int(v) & 0xFFFFFFFF for v in unk])
    return bytes(out)


def encode_geo0(geo0: dict, version: int) -> bytes:
    if not geo0:
        return b''
    if geo0.get('opaque'):
        return _b64d(geo0.get('raw_b64', ''))

    v = geo0.get('v') or {}
    vert_count = len(v.get('px', []))
    raw = _join_verts(v, 36)
    stored = _b64d(geo0.get('vb_comp_b64', ''))
    if raw is not None and geo0.get('vb_hash') and _md5(raw) == geo0['vb_hash'] and stored:
        vb = stored
    elif raw is not None and vert_count > 0:
        vb = meshopt_encode_vertex_buffer(vert_count, 36, raw)
    else:
        vb = stored

    ib = _b64d(geo0.get('ib_b64', ''))
    chunks = geo0.get('chunks', []) or []
    cloud_chunks = geo0.get('cloud_chunks', []) or []
    subchunks = geo0.get('subchunks', []) or []

    out = bytearray()
    out += struct.pack('<6I', len(ib), vert_count, len(chunks), len(cloud_chunks),
                       len(subchunks), len(vb))
    out += vb
    out += ib
    for chunk in chunks:
        out += _pack_chunk(chunk)
    for chunk in cloud_chunks:
        out += _pack_chunk(chunk)
    for sub in subchunks:
        out += struct.pack('<8B', int(sub.get('material_id', 0)) & 0xFF,
                           int(sub.get('triangle_count', 0)) & 0xFF,
                           int(sub.get('vtx_count', 0)) & 0xFF,
                           int(sub.get('triangle_start', 0)) & 0xFF,
                           int(sub.get('triangle_end', 0)) & 0xFF,
                           int(sub.get('vtx_start', 0)) & 0xFF,
                           int(sub.get('vtx_end', 0)) & 0xFF,
                           int(sub.get('unk', 0)) & 0xFF)
    out += _b64d(geo0.get('trailing_b64', ''))
    return bytes(out)


# ---- vertex helpers ------------------------------------------------------------
def _split_verts(raw: bytes, count: int, stride: int) -> dict:
    px, py, pz = [], [], []
    rest = bytearray()
    rest_len = stride - 12
    for i in range(count):
        offset = i * stride
        x, y, z = struct.unpack_from('<3f', raw, offset)
        px.append(x)
        py.append(y)
        pz.append(z)
        rest += raw[offset + 12:offset + stride]
    return {'px': px, 'py': py, 'pz': pz, 'rest_b64': _b64e(rest) if rest else '',
            'rest_len': rest_len}


def _join_verts(v: dict, stride: int):
    px = v.get('px') or []
    n = len(px)
    if n == 0:
        return b''
    py = v.get('py') or []
    pz = v.get('pz') or []
    if len(py) != n or len(pz) != n:
        return None
    rest = _b64d(v.get('rest_b64', ''))
    rest_len = stride - 12
    if len(rest) != n * rest_len:
        return None
    out = bytearray(n * stride)
    for i in range(n):
        struct.pack_into('<3f', out, i * stride, px[i], py[i], pz[i])
        out[i * stride + 12:(i + 1) * stride] = rest[i * rest_len:(i + 1) * rest_len]
    return bytes(out)


# ---- whole file ----------------------------------------------------------------
def decode_generic_meshes(stream: BinaryStreamReader) -> list:
    """Prop/palette list.  Same as the legacy reader but keeps the opaque four
    byte field that newer files store for separator==1 submeshes."""
    result = []
    total = stream.read_uint32()
    for i in range(total):
        mesh = {'index': i}
        name_len = stream.read_uint32()
        if name_len > 1024:
            stream.byte_pos -= 4
            break
        mesh['name'] = stream.read_bytes(name_len).decode('ascii', errors='replace')
        mesh['resource_id'] = stream.read_uint32()
        submesh_count = stream.read_uint32()
        mesh['submesh_count'] = submesh_count
        sep = stream.read_bytes(1)[0]
        mesh['separator'] = sep
        mesh['submeshes'] = []

        if sep == 0x00:
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            raw = stream.read_bytes(12 * cc)
            colors = []
            for b in range(cc):
                for q in range(3):
                    off = b * 12 + q * 4
                    colors.append([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
            mesh['submeshes'].append({
                'mode': 0, 'color_blocks': cc, 'gap': gg,
                'colors': colors, 'gap_data': list(stream.read_bytes(gg))})
        elif sep == 0x01:
            mode = 1
            mode_bytes = list(stream.read_bytes(4))
            gg = stream.read_uint32()
            raw = stream.read_bytes(12)
            colors = []
            for q in range(3):
                off = q * 4
                colors.append([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
            mesh['submeshes'].append({
                'mode': 1, 'mode_bytes': mode_bytes, 'color_blocks': 1, 'gap': gg,
                'colors': colors, 'gap_data': list(stream.read_bytes(gg))})
        else:
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            raw = stream.read_bytes(12 * cc)
            colors = []
            for b in range(cc):
                for q in range(3):
                    off = b * 12 + q * 4
                    colors.append([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
            mesh['submeshes'].append({
                'mode': 0, 'color_blocks': cc, 'gap': gg,
                'colors': colors, 'gap_data': list(stream.read_bytes(gg))})

        for _j in range(1, submesh_count):
            if mode == 0:
                cc = stream.read_uint32()
                gg = stream.read_uint32()
                raw = stream.read_bytes(12 * cc)
                colors = []
                for b in range(cc):
                    for q in range(3):
                        off = b * 12 + q * 4
                        colors.append([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
                mesh['submeshes'].append({
                    'mode': 0, 'color_blocks': cc, 'gap': gg,
                    'colors': colors, 'gap_data': list(stream.read_bytes(gg))})
            else:
                mode_bytes = list(stream.read_bytes(4))
                gg = stream.read_uint32()
                raw = stream.read_bytes(12)
                colors = []
                for q in range(3):
                    off = q * 4
                    colors.append([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
                mesh['submeshes'].append({
                    'mode': 1, 'mode_bytes': mode_bytes, 'color_blocks': 1, 'gap': gg,
                    'colors': colors, 'gap_data': list(stream.read_bytes(gg))})

        result.append(mesh)
    return result


def encode_generic_meshes(writer: BinaryStreamWriter, meshes: list):
    writer.write_uint32(len(meshes))
    for m in meshes:
        name_bytes = m['name'].encode('ascii')
        writer.write_uint32(len(name_bytes))
        writer.write_bytes(name_bytes)
        writer.write_uint32(m['resource_id'])
        writer.write_uint32(m['submesh_count'])
        sep = m['separator']
        writer.write_bytes(bytes([sep]))
        mode = m['submeshes'][0]['mode']
        for sm in m['submeshes']:
            if mode == 0:
                writer.write_uint32(sm['color_blocks'])
                writer.write_uint32(sm['gap'])
                raw = bytearray()
                for b in range(sm['color_blocks']):
                    for q in range(3):
                        raw.extend(bytes(sm['colors'][b * 3 + q]))
                writer.write_bytes(bytes(raw))
                writer.write_bytes(bytes(sm['gap_data']))
            else:
                mb = sm.get('mode_bytes') or [0, 0, 0, 0]
                writer.write_bytes(bytes(mb[:4]))
                writer.write_uint32(sm['gap'])
                raw = bytearray()
                for q in range(3):
                    raw.extend(bytes(sm['colors'][q]))
                writer.write_bytes(bytes(raw))
                writer.write_bytes(bytes(sm['gap_data']))


def decode_meshes_bytes(data: bytes) -> dict:
    """Decode a complete .meshes file (LVL0 container) into a JSON-ready dict."""
    if len(data) < 12 or data[0:4] != b'LVL0':
        # legacy raw payload (e.g. a .bin produced by the old decompressor)
        decoded = decode_lod0(data, 0x36)
        decoded.update({
            'format': 'sky-meshes-v2', 'version': 0x36, 'version_hex': '0x36',
            'container': {'raw_payload': True},
            'geo0': None, 'metr': None,
            'bounds': {'min': [0.0, 0.0, 0.0], 'max': [0.0, 0.0, 0.0]},
        })
        return decoded

    container = parse_container(data)
    version = container['version']
    payloads = read_section_payloads(data, container)

    lod_raw = None
    if 'LOD0' in payloads:
        try:
            lod_raw = _lz4_decompress_any(payloads['LOD0'])
        except MeshesFormatError:
            lod_raw = None
    if lod_raw is not None:
        decoded = decode_lod0(lod_raw, version)
    else:
        decoded = {'generic_meshes': [], 'terrain_patches': [], 'cloud': {'present': False},
                   'cloud_mesh': {'groups': [], 'mesh_16': None}}
        decoded['lod0_undecoded'] = True

    decoded['format'] = 'sky-meshes-v2'
    decoded['version'] = version
    decoded['version_hex'] = '0x%02X' % version
    decoded['version_hint'] = _version_name(version)
    bounds = container['bounds']
    decoded['bounds'] = {'min': bounds[0:3], 'max': bounds[3:6]}
    decoded['container'] = {
        'pad_hex': container['pad'].hex(),
        'gap_b64': _b64e(container['gap']),
        'order': [entry['name_str'] for entry in container['entries']],
    }
    if lod_raw is not None:
        decoded['container']['lod0'] = {
            'compressed_b64': _b64e(payloads['LOD0']),
            'raw_md5': _md5(lod_raw),
        }
    else:
        decoded['container']['lod0_raw_b64'] = _b64e(payloads['LOD0'])
    decoded['geo0'] = decode_geo0(payloads['GEO0'], version) if 'GEO0' in payloads else None
    decoded['metr'] = _b64e(payloads['METR']) if 'METR' in payloads else None
    extra = {name: _b64e(payload) for name, payload in payloads.items()
             if name not in ('LOD0', 'GEO0', 'METR')}
    if extra:
        decoded['container']['extra_sections'] = extra
    return decoded


def encode_meshes_json(decoded: dict, force_recompress: bool = False) -> bytes:
    """Encode a decoded dict back into a .meshes file."""
    version = int(decoded.get('version', 0x36))
    container = decoded.get('container', {}) or {}
    if decoded.get('lod0_undecoded') and container.get('lod0_raw_b64'):
        lod_comp = _b64d(container['lod0_raw_b64'])
        stored = {}
        lod_raw = None
    else:
        lod_raw = encode_lod0(decoded, version)
        stored = container.get('lod0') or {}
    if lod_raw is None:
        pass
    elif (not force_recompress and stored.get('raw_md5') == _md5(lod_raw)
            and stored.get('compressed_b64')):
        lod_comp = _b64d(stored['compressed_b64'])
    else:
        # Only fresh/edited output is constrained here; cached originals remain
        # byte-exact recoverable even if a historical asset exceeds this limit.
        _tr_check_lod0_limit(lod_raw)
        lod_comp = _lz4_compress(lod_raw)

    order = container.get('order') or default_section_order(decoded)
    extra = container.get('extra_sections') or {}
    sections = []
    for name in order:
        if name == 'LOD0':
            payload = lod_comp
        elif name == 'GEO0':
            payload = encode_geo0(decoded.get('geo0'), version)
        elif name == 'METR':
            payload = _b64d(decoded.get('metr') or '')
        else:
            payload = _b64d(extra.get(name, ''))
        if payload is None:
            continue
        sections.append((name, payload))

    bounds = decoded.get('bounds') or {}
    bounds6 = None
    if bounds.get('min') and bounds.get('max'):
        bounds6 = list(bounds['min'])[:3] + list(bounds['max'])[:3]
    if bounds6 is None:
        bounds6 = list(_compute_terrain_bbox(decoded))
    pad = bytes.fromhex(container.get('pad_hex', '')) if container.get('pad_hex') else None
    gap = _b64d(container.get('gap_b64', '')) if container.get('gap_b64') else None
    return build_container(version, sections, bounds6, pad=pad, gap=gap)


def meshes_to_json(input_path: str, output_path: str):
    with open(input_path, 'rb') as handle:
        data = handle.read()
    decoded = decode_meshes_bytes(data)
    decoded['_meta'] = {
        'source': os.path.basename(input_path),
        'total_bytes': len(data),
        'decoder': 'universal-v2',
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(decoded, handle, indent=2, ensure_ascii=False)
    print('JSON saved: %s  (version %s, %d bytes)' %
          (output_path, decoded.get('version_hex'), len(data)))
    return decoded


def json_to_meshes(json_path: str, output_path: str, as_lvl: bool = True):
    with open(json_path, 'r', encoding='utf-8') as handle:
        decoded = json.load(handle)
    raw = encode_meshes_json(decoded)
    with open(output_path, 'wb') as handle:
        handle.write(raw)
    print('meshes written: %s (%d bytes, version %s)' %
          (output_path, len(raw), decoded.get('version_hex', '?')))
    return output_path


def _compute_terrain_bbox(decoded: dict) -> tuple:
    """Fallback bounds: terrain vertices first, then GEO0 vertices."""
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for patch in decoded.get('terrain_patches', []) or []:
        v = patch.get('v', {})
        for px, py, pz in zip(v.get('px', []), v.get('py', []), v.get('pz', [])):
            min_x = min(min_x, px)
            min_y = min(min_y, py)
            min_z = min(min_z, pz)
            max_x = max(max_x, px)
            max_y = max(max_y, py)
            max_z = max(max_z, pz)

    geo0 = decoded.get('geo0') or {}
    v = geo0.get('v') or {}
    for px, py, pz in zip(v.get('px', []), v.get('py', []), v.get('pz', [])):
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        min_z = min(min_z, pz)
        max_x = max(max_x, px)
        max_y = max(max_y, py)
        max_z = max(max_z, pz)

    if min_x == float('inf'):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def json_to_obj(json_path: str, output_dir: str):
    """Export terrain + GEO0 + skirts + occluder triangles to a single OBJ."""
    with open(json_path, 'r', encoding='utf-8') as handle:
        decoded = json.load(handle)

    base = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    verts = []
    faces = []
    groups = []

    for patch in decoded.get('terrain_patches') or []:
        positions = patch.get('vertex_positions') or []
        triangles = patch.get('triangles') or []
        if not positions and patch.get('vertices'):
            positions = [v['pos'] for v in patch['vertices']]
        if not positions:
            continue
        start = len(verts)
        verts.extend(positions)
        for tri in triangles:
            faces.append((tri[0] + start, tri[1] + start, tri[2] + start))
        groups.append(('terrain_%d' % len(groups), start, len(faces)))

    geo0 = decoded.get('geo0') or {}
    if geo0 and not geo0.get('opaque'):
        v = geo0.get('v') or {}
        px, py, pz = v.get('px', []), v.get('py', []), v.get('pz', [])
        start = len(verts)
        verts.extend(zip(px, py, pz))
        ib = _b64d(geo0.get('ib_b64', ''))
        for chunk in geo0.get('chunks') or []:
            idx_start = int(chunk.get('idx_start', 0))
            idx_count = int(chunk.get('idx_count', 0))
            vtx_start = int(chunk.get('vtx_start', 0))
            seg = ib[idx_start:idx_start + idx_count]
            first = len(faces)
            for i in range(0, len(seg) - 2, 3):
                faces.append((start + vtx_start + seg[i],
                              start + vtx_start + seg[i + 1],
                              start + vtx_start + seg[i + 2]))
            if len(faces) > first:
                groups.append(('geo0_chunk_%d' % len(groups), start + vtx_start, first))

    for group in (decoded.get('cloud_mesh') or {}).get('groups') or []:
        indices = group.get('indices')
        v = group.get('v') or {}
        if not indices or not v.get('px'):
            continue
        start = len(verts)
        verts.extend(zip(v.get('px', []), v.get('py', []), v.get('pz', [])))
        first = len(faces)
        for i in range(0, len(indices) - 2, 3):
            faces.append((indices[i] + start, indices[i + 1] + start, indices[i + 2] + start))
        groups.append(('skirt_%d' % len(groups), start, first))

    mesh_16 = (decoded.get('cloud_mesh') or {}).get('mesh_16')
    if mesh_16:
        indices = mesh_16.get('indices') or []
        v = mesh_16.get('v') or {}
        if indices and v.get('px'):
            start = len(verts)
            verts.extend(zip(v.get('px', []), v.get('py', []), v.get('pz', [])))
            first = len(faces)
            for i in range(0, len(indices) - 2, 3):
                faces.append((indices[i] + start, indices[i + 1] + start, indices[i + 2] + start))
            groups.append(('occluder', start, first))

    obj_path = os.path.join(output_dir, '%s.obj' % base)
    with open(obj_path, 'w', encoding='ascii', errors='replace') as handle:
        handle.write('# %s\n' % base)
        handle.write('# version %s, %d vertices, %d triangles\n' %
                     (decoded.get('version_hex', '?'), len(verts), len(faces)))
        for v in verts:
            handle.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        group_index = 0
        for name, _start, first_face in groups:
            handle.write('g %s\n' % name)
            end = groups[group_index + 1][2] if group_index + 1 < len(groups) else len(faces)
            for tri in faces[first_face:end]:
                handle.write('f %d %d %d\n' % (tri[0] + 1, tri[1] + 1, tri[2] + 1))
            group_index += 1

    print('OBJ exported: %s (%dv, %dt)' % (obj_path, len(verts), len(faces)))
    return obj_path


# ---- guards for operations that only understand the legacy terrain layout ----
_legacy_merge_json = merge_json
_legacy_import_obj = import_obj_to_json


def _has_geo0(decoded: dict) -> bool:
    geo0 = decoded.get('geo0') or {}
    return bool(geo0.get('chunks') or geo0.get('raw_b64'))


def merge_json(json1: str, json2: str, output: str):
    with open(json1, 'r', encoding='utf-8') as fh:
        a = json.load(fh)
    with open(json2, 'r', encoding='utf-8') as fh:
        b = json.load(fh)
    if _has_geo0(a) or _has_geo0(b):
        raise MeshesFormatError(
            'merge is only implemented for terrain based levels (<= 0x37); '
            'GEO0 levels (0x38+) cannot be merged yet')
    if a.get('version') != b.get('version'):
        raise MeshesFormatError('cannot merge different versions: %s vs %s' %
                                (a.get('version_hex'), b.get('version_hex')))
    return _legacy_merge_json(json1, json2, output)


def import_obj_to_json(obj_path: str, json_path: str, output_json: str = None):
    with open(json_path, 'r', encoding='utf-8') as fh:
        decoded = json.load(fh)
    if _has_geo0(decoded):
        raise MeshesFormatError(
            'import-obj needs terrain patches to copy the vertex layout from; '
            'GEO0 levels (0x38+) are not supported - edit geo0.v positions in '
            'the JSON instead')
    return _legacy_import_obj(obj_path, json_path, output_json or json_path)


def meshes_info(path: str) -> dict:
    """Print a compact summary of a .meshes file without dumping the JSON."""
    with open(path, 'rb') as fh:
        data = fh.read()
    container = parse_container(data)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        decoded = decode_meshes_bytes(data)
    lod = decoded
    geo0 = decoded.get('geo0') or {}
    cloud = decoded.get('cloud') or {}
    cm = decoded.get('cloud_mesh') or {}
    info = {
        'file': os.path.basename(path),
        'bytes': len(data),
        'version': container['version'],
        'version_hex': '%#04x' % container['version'],
        'name': _version_name(container['version']),
        'sections': [(e['name_str'], e['offset'], e['length']) for e in container['entries']],
        'bounds': container['bounds'],
        'generic_meshes': len(lod.get('generic_meshes') or []),
        'terrain_patches': len(lod.get('terrain_patches') or []),
        'cloud_present': bool(cloud.get('present')),
        'cloud_dim': cloud.get('dim'),
        'skirts': len(cm.get('groups') or []),
        'occluder': bool(cm.get('mesh_16')),
        'geo0_vertices': geo0.get('vert_count', 0),
        'geo0_chunks': len(geo0.get('chunks') or []),
        'geo0_opaque': bool(geo0.get('opaque')),
        'lod_remaining': len(lod.get('_remaining') or b''),
    }
    for key, value in info.items():
        print('%-16s %s' % (key + ':', value))
    return info


# ---- section dumping helpers (compat with the old CLI) -------------------------
def parse_toc(data: bytes) -> list:
    container = parse_container(data)
    return [{'name': entry['name_str'], 'offset': entry['offset'], 'length': entry['length']}
            for entry in container['entries']]


def decompress_lvl(input_path: str, output_dir: str) -> list:
    with open(input_path, 'rb') as handle:
        data = handle.read()
    container = parse_container(data)
    payloads = read_section_payloads(data, container)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_paths = []
    for name, payload in payloads.items():
        if name == 'LOD0':
            raw = _lz4_decompress_any(payload)
        else:
            raw = payload
        out_path = os.path.join(output_dir, '%s_%s.bin' % (base_name, name))
        with open(out_path, 'wb') as handle:
            handle.write(raw)
        out_paths.append(out_path)
        print('  %s: %d -> %d bytes -> %s' % (name, len(payload), len(raw), out_path))
    return out_paths


def compress_lvl(decompressed_paths: list, output_path: str, base_name: str = 'LOD0',
                 version: int = 0x36):
    """Rebuild a container from dumped section files (LOD0 is re-compressed)."""
    sections = []
    for index, path in enumerate(decompressed_paths):
        with open(path, 'rb') as handle:
            raw = handle.read()
        name = 'LOD0' if index == 0 else os.path.splitext(os.path.basename(path))[0].split('_')[-1][:4]
        payload = _lz4_compress(raw) if name == 'LOD0' else raw
        sections.append((name, payload))
    data = build_container(version, sections, None)
    with open(output_path, 'wb') as handle:
        handle.write(data)
    print('LVL0 container written: %s (%d bytes)' % (output_path, len(data)))
    return output_path


def print_usage():
    print("""
Sky: Children of the Light - LVL/MESHES universal decoder/encoder/merger
=======================================================================
Supports every known .meshes version: 0x33 0x34 0x35 0x36 0x37 0x38 0x39
0x3A 0x3B 0x3C 0x3D (GEO0 + LOD0 + METR containers included).

USAGE:
  python Meshes-universal.py decode  input.meshes [output.json]
  python Meshes-universal.py info    input.meshes
  python Meshes-universal.py encode  input.json   [output.meshes]
  python Meshes-universal.py import-obj <obj> <json> [output.json]
  python Meshes-universal.py export-obj <json> [output_dir]
  python Meshes-universal.py merge   <json1> <json2> <output.json>
  python Meshes-universal.py help
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == 'decode':
        if len(sys.argv) < 3:
            print("Usage: python Meshes.py decode <input.meshes> [output.json]")
            sys.exit(1)
        inp = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(inp)[0] + '.json'
        meshes_to_json(inp, out)

    elif cmd == 'info':
        if len(sys.argv) < 3:
            print("Usage: python Meshes-universal.py info <input.meshes>")
            sys.exit(1)
        meshes_info(sys.argv[2])

    elif cmd == 'encode':
        if len(sys.argv) < 3:
            print("Usage: python Meshes.py encode <input.json> [output.meshes]")
            sys.exit(1)
        inp = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(inp)[0] + '.meshes'
        json_to_meshes(inp, out, as_lvl=True)

    elif cmd in ('import-obj', 'import_obj'):
        if len(sys.argv) < 4:
            print("Usage: python Meshes.py import-obj <obj_path> <json_path> [output.json]")
            sys.exit(1)
        out_json = sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]
        import_obj_to_json(sys.argv[2], sys.argv[3], out_json)

    elif cmd in ('export-obj', 'export_obj'):
        if len(sys.argv) < 3:
            print("Usage: python Meshes.py export-obj <json_path> [output_dir]")
            sys.exit(1)
        json_to_obj(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'obj_export')

    elif cmd == 'merge':
        if len(sys.argv) < 5:
            print("Usage: python Meshes.py merge <json1> <json2> <output.json>")
            sys.exit(1)
        merge_json(sys.argv[2], sys.argv[3], sys.argv[4])

    elif cmd in ('help', '-h', '--help'):
        print_usage()

    else:
        print("Unknown command: %s" % cmd)
        print_usage()
        sys.exit(1)


# ============================================================================
# JSON version translation (v3): real layouts, explicit losses, portable archive
# ============================================================================
import contextlib as _tr_contextlib
import io as _tr_io
import tempfile as _tr_tempfile
import zlib as _tr_zlib
from collections import Counter as _tr_Counter

# Representative loaders from 0.3.18 through 0.35.1 allocate this fixed-size
# decompression buffer. Unchanged original files can still roundtrip verbatim;
# newly encoded/translated LOD0 must fit the loader, not merely Python's decoder.
_TR_LOD0_LIMIT = 12 * 1024 * 1024
_TR_LEGACY_MATERIAL_IDS = frozenset((1, 2, 3, *range(16, 37), *range(48, 53), 80))

# The meshopt index decoder below follows meshoptimizer's MIT-licensed
# src/indexcodec.cpp, Copyright (c) 2016-2026 Arseny Kapoulkine.
# The complete license is retained in _MESHOPT_INDEX_LICENSE.
_MESHOPT_INDEX_LICENSE = """MIT License
Copyright (c) 2016-2026 Arseny Kapoulkine
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def parse_meshes_version(value):
    """Accept 0x33..0x3D, bare hexadecimal 33..3D, or decimal 51..61."""
    if isinstance(value, bool):
        raise MeshesFormatError('版本号不能是布尔值')
    if isinstance(value, int):
        version = value
    elif isinstance(value, str):
        text = value.strip().lower()
        try:
            if text.startswith('0x'):
                version = int(text, 16)
            elif text.isdecimal() and 51 <= int(text) <= 61:
                version = int(text)
            elif len(text) == 2 and text[0] == '3':
                version = int(text, 16)
            else:
                raise ValueError()
        except ValueError:
            raise MeshesFormatError('无效版本 %r；请输入 0x33～0x3D 或十进制 51～61' % value)
    else:
        raise MeshesFormatError('无效版本号类型')
    if not SUPPORTED_VERSION_MIN <= version <= SUPPORTED_VERSION_MAX:
        raise MeshesFormatError('不支持版本 %#x；支持 0x33～0x3D' % version)
    return version


def _tr_require(condition, message):
    if not condition:
        raise MeshesFormatError(message)


def _tr_u32_bytes(values):
    out = array.array('I', values)
    if sys.byteorder != 'little':
        out.byteswap()
    return out.tobytes()


def _tr_read_varint(data, pos, end):
    value = 0
    for shift in range(0, 35, 7):
        _tr_require(pos < end, 'meshopt 索引变长整数截断')
        byte = data[pos]
        pos += 1
        if shift == 28:
            _tr_require(byte < 16, 'meshopt 索引变长整数溢出')
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise MeshesFormatError('meshopt 索引变长整数过长')


def _tr_write_varint(out, value):
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def meshopt_decode_index_buffer(index_count, data, index_size=2):
    """Decode E0/E1 triangle streams and D0/D1 index sequences, with bounds checks."""
    data = bytes(data)
    _tr_require(isinstance(index_count, int) and 0 <= index_count <= 30_000_000,
                'meshopt 索引数量不合理')
    _tr_require(index_size in (2, 4), '索引大小必须为 2 或 4')
    if not data and index_count == 0:
        return []
    _tr_require(bool(data), 'meshopt 索引流为空')
    tag, version = data[0] & 0xF0, data[0] & 15
    _tr_require(tag in (0xD0, 0xE0) and version <= 1, '未知 meshopt 索引头')
    maximum = (1 << (index_size * 8)) - 1
    if tag == 0xD0:
        _tr_require(len(data) >= 1 + index_count + 4, 'meshopt 索引序列截断')
        pos, end, last, result = 1, len(data) - 4, [0, 0], []
        for _ in range(index_count):
            code, pos = _tr_read_varint(data, pos, end)
            which, delta = code & 1, code >> 1
            value = (last[which] + ((delta >> 1) ^ -(delta & 1))) & 0xFFFFFFFF
            _tr_require(value <= maximum, 'meshopt 索引超出目标位宽')
            last[which] = value
            result.append(value)
        _tr_require(pos == end, 'meshopt 索引序列存在未消费数据')
        return result

    _tr_require(index_count % 3 == 0, '三角形索引数必须为 3 的倍数')
    triangles = index_count // 3
    _tr_require(len(data) >= 1 + triangles + 16, 'meshopt 三角形索引流截断')
    pos, end = 1 + triangles, len(data) - 16
    table = data[end:]
    _tr_require(all((x & 15) != 15 and (x >> 4) != 15 for x in table),
                'meshopt 索引辅助表包含非法引用')
    edges, vertices = [(0xFFFFFFFF, 0xFFFFFFFF)] * 16, [0xFFFFFFFF] * 16
    edge_pos = vertex_pos = next_vertex = last_vertex = 0
    fec_max = 13 if version else 15
    result = []

    def free_index():
        nonlocal pos, last_vertex
        code, pos = _tr_read_varint(data, pos, end)
        last_vertex = (last_vertex + ((code >> 1) ^ -(code & 1))) & 0xFFFFFFFF
        return last_vertex

    def push_vertex(value):
        nonlocal vertex_pos
        vertices[vertex_pos] = value
        vertex_pos = (vertex_pos + 1) & 15

    def push_edge(a, b):
        nonlocal edge_pos
        edges[edge_pos] = (a, b)
        edge_pos = (edge_pos + 1) & 15

    for triangle in range(triangles):
        code = data[1 + triangle]
        if code < 0xF0:
            a, b = edges[(edge_pos - 1 - (code >> 4)) & 15]
            fec = code & 15
            if fec < fec_max:
                c = next_vertex if fec == 0 else vertices[(vertex_pos - 1 - fec) & 15]
                if fec == 0:
                    next_vertex += 1
                    push_vertex(c)
            else:
                if fec == 15:
                    c = free_index()
                else:
                    last_vertex = (last_vertex + fec * 2 - 27) & 0xFFFFFFFF
                    c = last_vertex
                push_vertex(c)
            push_edge(c, b)
            push_edge(a, c)
        elif code < 0xFE:
            aux = table[code & 15]
            feb, fec = aux >> 4, aux & 15
            a = next_vertex
            next_vertex += 1
            b = next_vertex if feb == 0 else vertices[(vertex_pos - feb) & 15]
            next_vertex += feb == 0
            c = next_vertex if fec == 0 else vertices[(vertex_pos - fec) & 15]
            next_vertex += fec == 0
            push_vertex(a)
            if feb == 0:
                push_vertex(b)
            if fec == 0:
                push_vertex(c)
            push_edge(b, a)
            push_edge(c, b)
            push_edge(a, c)
        else:
            _tr_require(pos < end, 'meshopt 索引辅助字节截断')
            aux = data[pos]
            pos += 1
            if aux == 0:
                next_vertex = 0
            fea, feb, fec = (0 if code == 0xFE else 15), aux >> 4, aux & 15
            a = next_vertex if fea == 0 else 0
            next_vertex += fea == 0
            b = next_vertex if feb == 0 else vertices[(vertex_pos - feb) & 15]
            next_vertex += feb == 0
            c = next_vertex if fec == 0 else vertices[(vertex_pos - fec) & 15]
            next_vertex += fec == 0
            if fea == 15:
                a = free_index()
            if feb == 15:
                b = free_index()
            if fec == 15:
                c = free_index()
            push_vertex(a)
            if feb in (0, 15):
                push_vertex(b)
            if fec in (0, 15):
                push_vertex(c)
            push_edge(b, a)
            push_edge(c, b)
            push_edge(a, c)
        _tr_require(max(a, b, c) <= maximum, 'meshopt 索引引用未初始化 FIFO 或超出位宽')
        result.extend((a, b, c))
    _tr_require(pos == end, 'meshopt 索引流存在未消费数据')
    return result


def meshopt_encode_index_buffer(indices, version=0):
    """Portable, deterministic E0/E1 encoder. Free-index codes preserve exact order.

    This deliberately favors correctness over compression ratio. No Node, compiler,
    native DLL, vertex remapping, or triangle rotations are required.
    """
    _tr_require(version in (0, 1), 'meshopt 索引编码版本必须为 0 或 1')
    _tr_require(len(indices) % 3 == 0 and len(indices) <= 30_000_000,
                '三角形索引数不合法')
    _tr_require(all(isinstance(x, int) and not isinstance(x, bool) and
                    0 <= x <= 0xFFFFFFFF for x in indices), '索引必须是 uint32 整数')
    out = bytearray([0xE0 | version])
    out.extend(b'\xFF' * (len(indices) // 3))
    last = 0
    for at in range(0, len(indices), 3):
        out.append(0xFF)
        for index in indices[at:at + 3]:
            delta = (index - last) & 0xFFFFFFFF
            signed = delta if delta < 0x80000000 else delta - 0x100000000
            zigzag = ((signed << 1) ^ (signed >> 31)) & 0xFFFFFFFF
            _tr_write_varint(out, zigzag)
            last = index
    out.extend(b'\x00' * 16)
    return bytes(out)


def _tr_decode_ib(target, count_key='idx_count', size=2):
    try:
        indices = meshopt_decode_index_buffer(int(target.get(count_key, 0)),
                                              _b64d(target.get('ib_b64', '')), size)
        target['indices'] = indices
        target['ib_hash'] = _md5(_tr_u32_bytes(indices))
    except (MeshesFormatError, ValueError, TypeError) as exc:
        target['ib_error'] = str(exc)
    return target


def _tr_render_ib(target, count_key='idx_count', size=2):
    stored = _b64d(target.get('ib_b64', ''))
    if 'indices' not in target or target.get('ib_error'):
        return stored, int(target.get(count_key, 0))
    indices = target['indices']
    _tr_require(isinstance(indices, list), 'indices 必须为列表')
    _tr_require(all(isinstance(x, int) and not isinstance(x, bool) and
                    0 <= x < (1 << (size * 8)) for x in indices), '索引超出目标位宽')
    if target.get('ib_hash') == _md5(_tr_u32_bytes(indices)) and stored:
        return stored, len(indices)
    return (meshopt_encode_index_buffer(indices) if indices else b''), len(indices)


_decode_meshopt_skirt_v2 = _decode_meshopt_skirt
_encode_cloud_mesh_v2 = encode_cloud_mesh
_decode_geo0_v2 = decode_geo0
_encode_geo0_v2 = encode_geo0
_encode_meshes_json_v2 = encode_meshes_json


def _decode_meshopt_skirt(stream, version):
    group = _decode_meshopt_skirt_v2(stream, version)
    if group.get('vert_count', 0) > 0 and group.get('vb_size', 0) == 0:
        group['vb_error'] = '非零顶点数但没有顶点缓冲；原样保留，禁止跨版本转译'
    return _tr_decode_ib(group)


def encode_cloud_mesh(writer, cloud_mesh, version=0x36):
    if version <= 0x36:
        return _encode_cloud_mesh_v2(writer, cloud_mesh, version)
    groups = cloud_mesh.get('groups') or []
    writer.write_uint32(len(groups))
    for group in groups:
        stride = 40 if version <= 0x38 else 36
        count = (len(group['v']['px']) if not group.get('vb_error') and
                 group.get('v') is not None else int(group.get('vert_count', 0)))
        writer.write_uint32(count)
        _encode_meshopt_vb(writer, group, stride)
        ib, index_count = _tr_render_ib(group)
        writer.write_uint32(index_count)
        writer.write_uint32(len(ib))
        writer.write_bytes(ib)
    # The occluder layout is common to all versions; reuse its existing writer,
    # omitting only the empty skirt-count prefix in that temporary buffer.
    tail = BinaryStreamWriter()
    _encode_cloud_mesh_v2(tail, {'groups': [], 'mesh_16': cloud_mesh.get('mesh_16')}, version)
    writer.write_bytes(tail.to_bytes()[4:])


def _tr_decode_geo038(data):
    """0x38: 5-u32 header, optional meshopt VB/IB, material draw groups.

    Header = vertex_stride, vertex_count, index_stride, index_count, group_count.
    Each group = u32 record_count, u32 material_id, record_count * 48 bytes.
    Each record = u32 index_start/count, AABB (6f), 4 preserved unknown u32.
    Material 0 is the base/depth pass; material 80 is the cloud render pass.
    """
    _tr_require(len(data) >= 20, '0x38 GEO0 头部截断')
    stride, count, index_stride, index_count, group_count = struct.unpack_from('<5I', data)
    _tr_require(stride == 36 and index_stride in (2, 4), '未知 0x38 GEO0 顶点/索引布局')
    _tr_require(count <= 5_000_000 and index_count <= 30_000_000 and group_count <= 256,
                '0x38 GEO0 数量不合理')
    reader = BinaryStreamReader(data)
    reader.byte_pos = 20
    result = {'layout': 'geo0-0x38', 'stride': stride, 'index_stride': index_stride,
              'vert_count': count, 'idx_count': index_count, 'draw_groups': []}
    result['vb_size'] = reader.read_uint32() if count else 0
    _tr_require(not count or result['vb_size'] > 0, '0x38 非零顶点数但无顶点缓冲')
    _decode_meshopt_vb(reader, result, count, stride, 'GEO0')
    result['ib_size'] = reader.read_uint32() if index_count else 0
    result['ib_b64'] = _b64e(reader.read_bytes(result['ib_size']))
    _tr_decode_ib(result, size=index_stride)
    for _ in range(group_count):
        records, material = reader.read_uint32(), reader.read_uint32()
        _tr_require(records <= reader.remaining() // 48, '0x38 GEO0 描述表截断')
        group = {'material_id': material, 'records': []}
        for _ in range(records):
            row = reader.read_bytes(48)
            start, length = struct.unpack_from('<2I', row)
            _tr_require(start + length <= index_count and length % 3 == 0,
                        '0x38 GEO0 描述表索引范围无效')
            group['records'].append({'idx_start': start, 'idx_count': length,
                                     'min': list(struct.unpack_from('<3f', row, 8)),
                                     'max': list(struct.unpack_from('<3f', row, 20)),
                                     'unk': list(struct.unpack_from('<4I', row, 32))})
        result['draw_groups'].append(group)
    if reader.remaining():
        result['trailing_b64'] = _b64e(reader.read_bytes(reader.remaining()))
    return result


def decode_geo0(data, version):
    if not data:
        return None
    if version == 0x38:
        try:
            return _tr_decode_geo038(data)
        except (EOFError, MeshesFormatError, struct.error) as exc:
            return {'opaque': True, 'raw_b64': _b64e(data),
                    'note': '0x38 GEO0 未知布局，原样保留：%s' % exc}
    if version >= 0x39 and len(data) == 20 and not any(struct.unpack('<5I', data)):
        return {'empty_header20': True, 'vert_count': 0,
                'v': _split_verts(b'', 0, 36), 'chunks': [], 'cloud_chunks': [],
                'subchunks': [], 'ib_b64': '', 'vb_comp_b64': '', 'vb_hash': _md5(b'')}
    result = _decode_geo0_v2(data, version)
    if result and not result.get('opaque') and result.get('vert_count') == 0:
        if not _b64d(result.get('vb_comp_b64', '')):
            result.pop('vb_error', None)
            result['v'] = _split_verts(b'', 0, 36)
            result['vb_hash'] = _md5(b'')
    return result


def encode_geo0(geo0, version):
    if geo0 and geo0.get('layout') == 'geo0-0x38':
        _tr_require(version == 0x38, '0x38 GEO0 不能直接写入其他版本；请使用功能 3 转译')
        count = (int(geo0.get('vert_count', 0)) if geo0.get('vb_error') else
                 len((geo0.get('v') or {}).get('px', [])))
        vb = render_meshopt_vb(geo0, 36)
        ib, index_count = _tr_render_ib(geo0, size=int(geo0.get('index_stride', 4)))
        groups = geo0.get('draw_groups') or []
        out = bytearray(struct.pack('<5I', 36, count, int(geo0.get('index_stride', 4)),
                                    index_count, len(groups)))
        if count:
            out.extend(struct.pack('<I', len(vb)))
            out.extend(vb)
        if index_count:
            out.extend(struct.pack('<I', len(ib)))
            out.extend(ib)
        for group in groups:
            records = group.get('records') or []
            out.extend(struct.pack('<2I', len(records), int(group['material_id'])))
            for row in records:
                out.extend(struct.pack('<2I6f4I', int(row['idx_start']), int(row['idx_count']),
                                       *(list(row['min']) + list(row['max']) + list(row['unk']))))
        out.extend(_b64d(geo0.get('trailing_b64', '')))
        return bytes(out)
    if geo0 and geo0.get('empty_header20'):
        if not (geo0.get('v') or {}).get('px') and not _b64d(geo0.get('ib_b64', '')):
            return struct.pack('<5I', 0, 0, 0, 0, 0)
        geo0 = dict(geo0)
        geo0.pop('empty_header20', None)
    return _encode_geo0_v2(geo0, version)


def _tr_terrain_raw(patch):
    v = patch.get('v') or {}
    count = int(patch.get('vert_count', len(v.get('px', []))))
    names = ('px', 'py', 'pz', 'normal_u', 'normal_v', 'material_id', 'vertex_attr',
             'cr', 'cg', 'cb', 'ca', 'extra_a', 'extra_b')
    _tr_require(all(len(v.get(key, [])) == count for key in names), '旧地形顶点字段长度不一致')
    out = bytearray(count * 36)
    for i in range(count):
        struct.pack_into('<3f3I4B2I', out, i * 36,
                         v['px'][i], v['py'][i], v['pz'][i],
                         v['normal_u'][i] | (v['normal_v'][i] << 16),
                         v['material_id'][i], v['vertex_attr'][i],
                         v['cr'][i], v['cg'][i], v['cb'][i], v['ca'][i],
                         v['extra_a'][i], v['extra_b'][i])
    return bytes(out)


def _tr_terrain_indices(patch):
    if patch.get('raw_indices') or int(patch.get('index_count', 0)) == 0:
        indices = list(patch.get('raw_indices') or [])
    elif patch.get('count_B'):
        raw = struct.pack('<%df' % len(patch['array_B']), *patch['array_B'])
        codes, offsets = patch.get('array_C') or [], patch.get('array_A') or []
        indices, at, base = [], 0, 0
        for i, entry in enumerate(patch.get('elem_loops') or []):
            for _ in range(int(entry['loop'])):
                _tr_require(at < len(codes), '旧地形压缩索引代码截断')
                code = codes[at]
                off = 4 * base + 4 * (code >> 1) + (2 if code & 1 else 0)
                _tr_require(off + 2 <= len(raw), '旧地形压缩索引偏移越界')
                indices.append(struct.unpack_from('<H', raw, off)[0])
                at += 1
            base += offsets[i] if i < len(offsets) else 0
    else:
        raise MeshesFormatError('旧地形没有可用的原始/压缩索引；拒绝仅使用可能已过滤的 triangles')
    _tr_require(len(indices) == int(patch.get('index_count', len(indices))),
                '旧地形 index_count 与解码索引数量不符')
    return indices


def _tr_check_surface(raw, stride, indices, label, check_positions=True):
    _tr_require(stride in (36, 40) and len(raw) % stride == 0, label + ' 顶点步长错误')
    count = len(raw) // stride
    _tr_require(len(indices) % 3 == 0, label + ' 三角形索引不完整')
    _tr_require(all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < count
                    for x in indices), label + ' 索引越界')
    if check_positions:
        for i in range(count):
            _tr_require(all(math.isfinite(x) for x in struct.unpack_from('<3f', raw, i * stride)),
                        label + ' 含非有限坐标')
    return count


def _tr_skirt_surface(group, version, label):
    if version <= 0x36:
        writer = BinaryStreamWriter()
        _encode_raw_skirt(writer, group)
        packed = writer.to_bytes()
        count = int(group.get('vert_count', 0))
        raw, stride = packed[4:4 + count * 40], 40
        indices = list(group.get('indices') or [])
        _tr_require(len(indices) == int(group.get('idx_count', len(indices))),
                    label + ' 索引数量不符')
    else:
        stride = 40 if version <= 0x38 else 36
        _tr_require(not group.get('vb_error') and not group.get('ib_error'),
                    label + ' 顶点/索引未解码，无法安全转译')
        raw = _join_verts(group.get('v') or {}, stride)
        _tr_require(raw is not None, label + ' 顶点字段不完整')
        indices = list(group['indices']) if 'indices' in group else meshopt_decode_index_buffer(
            int(group.get('idx_count', 0)), _b64d(group.get('ib_b64', '')), 2)
    _tr_check_surface(raw, stride, indices, label)
    return raw, stride, indices, label


def _tr_geo_surfaces(geo0, version):
    if not geo0:
        return []
    _tr_require(not geo0.get('opaque') and not geo0.get('vb_error') and not geo0.get('ib_error'),
                'GEO0 未完整解码；拒绝把不透明地形复制到其他版本')
    raw = _join_verts(geo0.get('v') or {}, 36)
    _tr_require(raw is not None, 'GEO0 顶点字段不完整')
    surfaces = []
    if geo0.get('layout') == 'geo0-0x38':
        indices = (list(geo0['indices']) if 'indices' in geo0 else
                   meshopt_decode_index_buffer(int(geo0.get('idx_count', 0)),
                                               _b64d(geo0.get('ib_b64', '')), 4))
        _tr_check_surface(raw, 36, indices, 'GEO0')
        groups = geo0.get('draw_groups') or []
        base = [g for g in groups if int(g['material_id']) == 0]
        cloud = [g for g in groups if int(g['material_id']) == 80]
        # Material passes may repeat the same triangle. Preserve one oriented
        # geometry copy, but keep source pass metadata in the portable archive.
        selected = base + cloud if base else groups
        seen = set()
        for group in selected:
            local = []
            for row in group['records']:
                start, count = int(row['idx_start']), int(row['idx_count'])
                _tr_require(start >= 0 and count >= 0 and start + count <= len(indices),
                            '0x38 GEO0 描述表索引越界')
                for at in range(start, start + count, 3):
                    tri = tuple(indices[at:at + 3])
                    _tr_require(len(tri) == 3, '0x38 GEO0 三角形截断')
                    key = min(tri, tri[1:] + tri[:1], tri[2:] + tri[:2])
                    if not base and key in seen:
                        continue
                    seen.add(key)
                    local.extend(tri)
            if local:
                surfaces.append((raw, 36, local, 'cloud' if group['material_id'] == 80 else 'terrain'))
        if indices:
            covered = set()
            for group in selected:
                for row in group['records']:
                    covered.update(range(int(row['idx_start']), int(row['idx_start']) + int(row['idx_count'])))
            # Duplicate material-pass ranges are allowed. Geometry outside all
            # descriptors is unreferenced baking data, not silently rendered.
            _tr_require(bool(covered), '0x38 GEO0 有索引但无可用绘制描述表')
        return surfaces
    _tr_check_surface(raw, 36, [], 'GEO0')
    ib = _b64d(geo0.get('ib_b64', ''))
    for category, chunks in (('terrain', geo0.get('chunks') or []),
                             ('cloud', geo0.get('cloud_chunks') or [])):
        for chunk in chunks:
            start, count, base = int(chunk['idx_start']), int(chunk['idx_count']), int(chunk['vtx_start'])
            _tr_require(start >= 0 and count >= 0 and start + count <= len(ib), 'GEO0 本地索引范围越界')
            indices = [base + x for x in ib[start:start + count]]
            _tr_check_surface(raw, 36, indices, 'GEO0 ' + category, check_positions=False)
            surfaces.append((raw, 36, indices, category))
    return surfaces

def _tr_reindex_surfaces(raw, stride, indices, label, limit=65535, index_limit=None):
    """Split uint16 meshes without losing/reversing triangles or bin counts."""
    _tr_check_surface(raw, stride, indices, label, check_positions=False)
    mapping, vertices, local = {}, bytearray(), []
    for at in range(0, len(indices), 3):
        triangle = indices[at:at + 3]
        new_count = sum(1 for x in set(triangle) if x not in mapping)
        if local and (len(mapping) + new_count > limit or
                      (index_limit is not None and len(local) + 3 > index_limit)):
            yield bytes(vertices), local
            mapping, vertices, local = {}, bytearray(), []
        for index in triangle:
            if index not in mapping:
                mapping[index] = len(mapping)
                vertices.extend(raw[index * stride:(index + 1) * stride])
            local.append(mapping[index])
    if local:
        yield bytes(vertices), local


def _tr_skirt_runtime_alpha(value, version):
    """Return the normal.w byte after the loader's legacy alpha conversion.

    0.35.1 sub_18C92AC (version <= 0x3A) uses
    int(clamp(int8(value) * 10, 0, 255) - 128). Versions 0x3B+ store that
    biased byte directly. This is a loader-byte model, not a shader guess.
    """
    if version > 0x3A:
        return value
    signed = value if value < 128 else value - 256
    return (min(max(signed * 10, 0), 255) - 128) & 255


def _tr_skirt_alpha_to_version(value, source_version, target_version):
    if (source_version <= 0x3A) == (target_version <= 0x3A):
        return value
    modern = _tr_skirt_runtime_alpha(value, source_version)
    if target_version > 0x3A:
        return modern
    opacity = modern ^ 128
    if opacity == 255:
        return 127  # fully opaque legacy sentinel; exact after clamping
    return (opacity + 5) // 10


def _tr_convert_skirt_alpha(raw, stride, source_version, target_version):
    if (source_version <= 0x3A) == (target_version <= 0x3A):
        return raw, 0
    out = bytearray(raw)
    alpha_offset = 39 if stride == 40 else 15
    quantized = 0
    for at in range(alpha_offset, len(raw), stride):
        value = _tr_skirt_alpha_to_version(raw[at], source_version, target_version)
        quantized += (_tr_skirt_runtime_alpha(value, target_version) !=
                      _tr_skirt_runtime_alpha(raw[at], source_version))
        out[at] = value
    return bytes(out), quantized


def _tr_convert_skirt_vertices(raw, source_stride, target_stride, source_is_skirt):
    if source_stride == target_stride:
        return raw
    count = len(raw) // source_stride
    out = bytearray(count * target_stride)
    for i in range(count):
        src, dst = i * source_stride, i * target_stride
        out[dst:dst + 36] = raw[src:src + 36]
        if target_stride == 40:
            # Old skirt visibility lives in userData1.w; old normal.w is not alpha.
            visibility = raw[src + 15] if source_is_skirt else 127
            out[dst + 15] = 0
            out[dst + 36:dst + 40] = bytes((0, 0, 0, visibility))
        else:
            # Modern skirt alpha = clamp(normal.w * .5 + .5, 0, 1).
            out[dst + 15] = raw[src + 39] if source_is_skirt else 127
    return bytes(out)


def _tr_make_skirt(raw, indices, version):
    stride = 40 if version <= 0x38 else 36
    count = _tr_check_surface(raw, stride, indices, '目标裙边')
    _tr_require(all(index <= 65535 for index in indices), '目标裙边索引超过 uint16 限制')
    if version <= 0x36:
        packed = (struct.pack('<I', count) + raw + struct.pack('<I', len(indices)) +
                  struct.pack('<%dH' % len(indices), *indices))
        return _decode_raw_skirt(BinaryStreamReader(packed))
    return {'vert_count': count, 'stride': stride, 'v': _split_verts(raw, count, stride),
            'vb_comp_b64': '', 'vb_hash': '', 'idx_count': len(indices),
            'ib_b64': '', 'ib_hash': '', 'indices': indices}


def _tr_geo_material_weights(raw, index):
    """Effective material weights used by 0.35.1 TerrainGeo (sub_2368D04).

    The four IDs at 16..19 and weights at 20..23 are byte lanes, not one
    material/attribute integer. Matching weights are summed then stored as u8.
    Material 0 has no terrain draw pass. Preserve the original vertex bytes.
    """
    at = index * 36
    weights = {}
    for material, weight in zip(raw[at + 16:at + 20], raw[at + 20:at + 24]):
        weights[material] = (weights.get(material, 0) + weight) & 255
    return {material: weight for material, weight in weights.items() if material and weight}


def _tr_validate_geo0_render_tables(geo0):
    """Check modern meshlets against the actual 0.35.1 material expansion.

    This is a structural/loading check, NOT an in-game runtime test. In
    particular it cannot check external shaders, textures or level objects.
    """
    _tr_require(geo0 and not geo0.get('opaque') and not geo0.get('vb_error'),
                '目标 GEO0 顶点未完整解码')
    raw = _join_verts(geo0.get('v') or {}, 36)
    _tr_require(raw is not None, '目标 GEO0 顶点字段不完整')
    _tr_check_surface(raw, 36, [], '目标 GEO0')
    ib = _b64d(geo0.get('ib_b64', ''))
    subs = geo0.get('subchunks') or []
    weights = {}
    counts = {'vertices': len(raw) // 36, 'terrain_meshlets': 0, 'cloud_meshlets': 0,
              'terrain_triangles': 0, 'cloud_triangles': 0, 'material_subchunks': 0,
              'material_pass_triangles': 0, 'cone_culling_disabled_meshlets': 0}
    for category, chunks in (('terrain', geo0.get('chunks') or []),
                             ('cloud', geo0.get('cloud_chunks') or [])):
        for chunk in chunks:
            start, base = int(chunk['idx_start']), int(chunk['vtx_start'])
            index_count, vertex_count = int(chunk['idx_count']), int(chunk['vtx_count'])
            sub_start, sub_count = int(chunk['area_start']), int(chunk['area_count'])
            _tr_require(0 < index_count <= 756 and index_count % 3 == 0 and
                        0 < vertex_count <= 252 and 0 < sub_count <= 255,
                        '目标 GEO0 分块数量超出 u8/252 三角形布局限制')
            _tr_require(start >= 0 and start + index_count <= len(ib) and base >= 0 and
                        (base + vertex_count) * 36 <= len(raw), '目标 GEO0 分块缓冲越界')
            _tr_require(sub_start >= 0 and sub_start + sub_count <= len(subs),
                        '目标 GEO0 材质子块范围越界')
            local = ib[start:start + index_count]
            _tr_require(all(x < vertex_count for x in local), '目标 GEO0 本地 uint8 索引越界')
            triangles = [local[at:at + 3] for at in range(0, index_count, 3)]
            covered = set()
            for sub in subs[sub_start:sub_start + sub_count]:
                material = int(sub['material_id'])
                _tr_require(0 < material < 255 and ((material == 80) == (category == 'cloud')),
                            '目标 GEO0 地形/云材质子块分类无效')
                first, last = int(sub['triangle_start']), int(sub['triangle_end'])
                vfirst, vlast = int(sub['vtx_start']), int(sub['vtx_end'])
                _tr_require(0 <= first <= last < len(triangles) and
                            0 <= vfirst <= vlast < vertex_count, '目标 GEO0 材质子块区间越界')
                active = []
                referenced = set()
                for at in range(first, last + 1):
                    triangle = triangles[at]
                    visible = False
                    for index in triangle:
                        if vfirst <= index <= vlast:
                            global_index = base + index
                            if global_index not in weights:
                                weights[global_index] = _tr_geo_material_weights(raw, global_index)
                            visible |= bool(weights[global_index].get(material, 0))
                    if visible:
                        active.append(at)
                        referenced.update(triangle)
                _tr_require(len(active) == int(sub['triangle_count']) and
                            len(referenced) == int(sub['vtx_count']),
                            '目标 GEO0 材质筛选后的三角形/顶点数不符（游戏会断言或不绘制）')
                # Only the weight lookup is restricted to vfirst..vlast. A
                # drawn triangle may also reference vertices outside this range;
                # TerrainGeo copies them with weight 0. Native 0.35.1 maps use
                # such ranges, so do not mistake them for out-of-range indices.
                covered.update(active)
                counts['material_subchunks'] += 1
                counts['material_pass_triangles'] += len(active)
            _tr_require(len(covered) == len(triangles), '目标 GEO0 存在无有效材质绘制的三角形')
            for axis in range(3):
                lo, hi = float(chunk['min'][axis]), float(chunk['max'][axis])
                _tr_require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
                            '目标 GEO0 分块包围盒无效')
                _tr_require(all(lo <= struct.unpack_from('<3f', raw, (base + i) * 36)[axis] <= hi
                                for i in range(vertex_count)), '目标 GEO0 分块包围盒未包含顶点')
            counts[category + '_meshlets'] += 1
            counts[category + '_triangles'] += len(triangles)
            if int((chunk.get('unk') or [0, 0, 0, 0])[3]) >> 24 == 127:
                counts['cone_culling_disabled_meshlets'] += 1
    counts['all_triangles_material_covered'] = True
    counts['in_game_runtime_tested'] = False
    return counts


def _tr_check_lod0_limit(raw):
    size = len(raw)
    _tr_require(size <= _TR_LOD0_LIMIT,
                '目标 LOD0 解压大小 %d 字节超过加载器的 12 MiB 上限（%d 字节）；'
                '拒绝生成游戏加载器无法接收的目标文件' % (size, _TR_LOD0_LIMIT))
    return {'raw_bytes': size, 'limit_bytes': _TR_LOD0_LIMIT, 'within_loader_limit': True}


def _tr_oriented_key(triangle):
    triangle = tuple(triangle)
    return min(triangle, triangle[1:] + triangle[:1], triangle[2:] + triangle[:2])


def _tr_surface_bounds(raw, stride, indices=None):
    referenced = set(indices) if indices is not None else range(len(raw) // stride)
    lo, hi = [float('inf')] * 3, [float('-inf')] * 3
    for index in referenced:
        point = struct.unpack_from('<3f', raw, index * stride)
        for axis, value in enumerate(point):
            lo[axis] = min(lo[axis], value)
            hi[axis] = max(hi[axis], value)
    _tr_require(lo[0] != float('inf'), '不能为无顶点的地形块生成包围盒')
    # Bounds are serialized as float32. Round OUTWARD (including large world
    # coordinates), rather than letting round-to-nearest shrink a tight AABB.
    def outward(value, lower):
        expanded = value + (-1 if lower else 1) * max(0.01, abs(value) * 1e-6)
        if abs(expanded) > 3.4028234663852886e38:
            return value
        return struct.unpack('<f', struct.pack('<f', expanded))[0]
    return [outward(x, True) for x in lo], [outward(x, False) for x in hi]


def _tr_prepare_terrain_surfaces(surfaces):
    prepared = []
    for raw, stride, indices, label in surfaces:
        _tr_require(stride == 36, '原生地形顶点必须为 36B')
        _tr_check_surface(raw, stride, indices, '源 ' + label)
        cache = {i: _tr_geo_material_weights(raw, i) for i in set(indices)}
        categories = {'terrain': [], 'cloud': []}
        for at in range(0, len(indices), 3):
            triangle = indices[at:at + 3]
            materials = set().union(*(cache[i] for i in triangle))
            _tr_require(materials and all(0 < mat < 255 for mat in materials),
                        '源地形有零权重/无效材质三角形，不能声明目标可绘制')
            _tr_require(80 not in materials or materials == {80},
                        '混合体积云/普通地形材质不能安全重建原生地形；拒绝静默丢失材质')
            categories['cloud' if materials == {80} else 'terrain'].extend(triangle)
        for category, local in categories.items():
            if local:
                prepared.append((raw, stride, local, category))
    return prepared


def _tr_make_legacy_terrain(surfaces, version):
    """Real 0x33..0x37 TerrainMesh, not LOD0 skirt substitution.

    LoadLod accepts raw uint16 indices when tessData.count_B == 0. CreateModel
    consumes the count_A spatial bins even on its static path, and treats <2
    bins as static. A single enclosing cell per compact mesh provides a valid
    sphere/AABB and material bin without fabricating dynamic tessellation data.
    External ID -1 intentionally resolves no source-version TerrainBlobGroup;
    flag2=True disables adding an unverified collision instance.
    """
    _tr_require(version <= 0x37, '旧 LOD0 地形重建需要 0x33～0x37 目标')
    patches = []
    for raw, stride, indices, category in _tr_prepare_terrain_surfaces(surfaces):
        # CreateModel stores each spatial-bin material indexCount in uint16;
        # limit total bin indices as well as local vertex IDs.
        for compact, local in _tr_reindex_surfaces(raw, stride, indices, category,
                                                  index_limit=65535):
            count = len(compact) // 36
            lo, hi = _tr_surface_bounds(compact, stride)
            edge = max(hi[axis] - lo[axis] for axis in range(3))
            _tr_require(math.isfinite(edge) and 0 < edge <= 3.4028234663852886e38,
                        '旧地形空间分块边长超出 float32 可表示范围')
            edge = struct.unpack('<f', struct.pack('<f', edge * (1 + 1e-6)))[0]
            fields = {key: [] for key in ('px', 'py', 'pz', 'normal_u', 'normal_v',
                       'material_id', 'vertex_attr', 'cr', 'cg', 'cb', 'ca', 'extra_a', 'extra_b')}
            for vertex in struct.iter_unpack('<3f3I4B2I', compact):
                x, y, z, normal, materials, weights, cr, cg, cb, ca, extra_a, extra_b = vertex
                values = (x, y, z, normal & 65535, normal >> 16, materials, weights,
                          cr, cg, cb, ca, extra_a, extra_b)
                for key, value in zip(fields, values):
                    fields[key].append(value)
            patch = {'index': len(patches), 'external_id': -1, 'flag1': False, 'flag2': True,
                     'row0': lo, 'row1': hi, 'vert_count': count, 'index_count': len(local),
                     'v': fields, 'some_count': 0, 'some_data': [],
                     'row2': lo, 'row3': [lo[axis] + edge for axis in range(3)],
                     'float_value': edge, 'uint_116': 1, 'uint_120': 1, 'uint_124': 1,
                     'count_A': 1, 'elem_loops': [{'loop': len(local), 'value': 0}],
                     'count_A_repeat': 0, 'count_B': 0, 'count_C': 0,
                     'array_A': [], 'array_B': [], 'array_C': [], 'raw_indices': local}
            patches.append(patch)
    _tr_validate_legacy_terrain(patches, version)
    return patches


def _tr_legacy_vertex_materials(raw, index):
    """Default legacy MaterialBinify: ordered lanes, zero-ID terminator.

    Representative MaterialDefs initialize (minimum, full) thresholds to
    (0,255). A full lane stops scanning later IDs, unlike modern summed lanes.
    Refuse incompatible lane arrangements rather than emitting invisible faces.
    """
    at, active = index * 36, set()
    for material, weight in zip(raw[at + 16:at + 20], raw[at + 20:at + 24]):
        if material == 0:
            break
        if weight > 0:
            active.add(material)
        if weight == 255:
            break
    return active


def _tr_validate_legacy_terrain(patches, version):
    """Validate newly rebuilt static meshes, NOT arbitrary native dynamic maps."""
    counts = {'patches': len(patches), 'vertices': 0, 'terrain_triangles': 0,
              'cloud_triangles': 0, 'spatial_bins': 0, 'collision_instances_disabled': True}
    allowed = _TR_LEGACY_MATERIAL_IDS | ({4, 37} if version >= 0x36 else set())
    materials = set()
    for patch in patches:
        raw, indices = _tr_terrain_raw(patch), _tr_terrain_indices(patch)
        count = _tr_check_surface(raw, 36, indices, '目标旧地形')
        _tr_require(0 < count <= 65535 and 0 < len(indices) <= 65535,
                    '目标旧地形超出 uint16 顶点/空间材质分块索引数量限制')
        _tr_require(patch['external_id'] == -1 and patch['flag2'],
                    '重建旧地形不得引用源 TerrainBlobGroup 或声明已添加碰撞')
        _tr_require(patch['count_B'] == patch['count_C'] == patch['count_A_repeat'] == 0 and
                    patch['count_A'] == 1 and patch['elem_loops'] == [{'loop': len(indices), 'value': 0}],
                    '目标静态地形必须有完整的原始索引及单空间分块')
        _tr_require([patch[key] for key in ('uint_116', 'uint_120', 'uint_124')] == [1, 1, 1],
                    '目标静态地形空间网格维度无效')
        edge = float(patch['float_value'])
        _tr_require(math.isfinite(edge) and edge > 0, '目标静态地形空间单元边长无效')
        cache = {}
        for index in range(count):
            point = struct.unpack_from('<3f', raw, index * 36)
            _tr_require(all(patch['row0'][axis] <= point[axis] <= patch['row1'][axis] and
                            patch['row2'][axis] <= point[axis] <= patch['row2'][axis] + edge
                            for axis in range(3)), '目标旧地形包围盒/空间球未覆盖所有顶点')
            ids = {mat for mat in raw[index * 36 + 16:index * 36 + 20] if mat}
            _tr_require(ids <= allowed,
                        '目标代表旧版未登记地形材质 ID：%s；不能用裙边掩盖材质不兼容' % sorted(ids - allowed))
            cache[index] = _tr_geo_material_weights(raw, index)
            _tr_require(_tr_legacy_vertex_materials(raw, index) == set(cache[index]),
                        '目标旧版有序材质槽筛选与源有效权重不符；拒绝不可绘制的静态转译')
            materials.update(ids)
        for at in range(0, len(indices), 3):
            active = set().union(*(cache[index] for index in indices[at:at + 3]))
            _tr_require(active and (80 not in active or active == {80}),
                        '目标旧地形三角形没有有效材质或混合云材质')
            counts['cloud_triangles' if active == {80} else 'terrain_triangles'] += 1
        counts['vertices'] += count
        counts['spatial_bins'] += 1
    counts.update({'registered_material_ids': sorted(materials), 'raw_uint16_indices': True,
                   'static_geometry': True, 'in_game_runtime_tested': False})
    return counts


def _tr_make_geo038(surfaces):
    """0x38 global uint32 meshopt indices with independent real material passes.

    Material 0 is the non-cloud depth/collision base; 80 is cloud; each other
    material receives exactly those oriented triangles with nonzero weight.
    Records are deliberately <=500 faces / <=1500 indices, leaving the loader's
    uint16 draw-start/index-count packing valid. Cone byte 127 disables culling.
    """
    vertices, indices, rows = bytearray(), [], {}
    for compact_source, stride, source_indices, category in _tr_prepare_terrain_surfaces(surfaces):
        for compact, local in _tr_reindex_surfaces(compact_source, stride, source_indices, category):
            base = len(vertices) // stride
            vertices.extend(compact)
            passes = {80: []} if category == 'cloud' else {0: []}
            cache = {index: _tr_geo_material_weights(compact, index) for index in set(local)}
            for at in range(0, len(local), 3):
                triangle = [base + index for index in local[at:at + 3]]
                active = set().union(*(cache[index] for index in local[at:at + 3]))
                passes[80 if category == 'cloud' else 0].extend(triangle)
                if category != 'cloud':
                    for material in sorted(active):
                        passes.setdefault(material, []).extend(triangle)
            for material, pass_indices in passes.items():
                for at in range(0, len(pass_indices), 1500):
                    piece = pass_indices[at:at + 1500]
                    start = len(indices)
                    indices.extend(piece)
                    local_indices = [index - base for index in piece]
                    lo, hi = _tr_surface_bounds(compact, stride, local_indices)
                    rows.setdefault(material, []).append({'idx_start': start, 'idx_count': len(piece),
                                                         'min': lo, 'max': hi,
                                                         'unk': [0, 0, 0, 0x7F000000]})
    result = {'layout': 'geo0-0x38', 'stride': 36, 'index_stride': 4,
              'vert_count': len(vertices) // 36, 'idx_count': len(indices),
              'v': _split_verts(bytes(vertices), len(vertices) // 36, 36),
              'vb_comp_b64': '', 'vb_hash': '', 'indices': indices,
              'ib_b64': '', 'ib_hash': '',
              'draw_groups': [{'material_id': material, 'records': rows[material]}
                              for material in sorted(rows, key=lambda mat: (mat not in (0, 80), mat))]}
    _tr_validate_geo038_render_tables(result)
    return result


def _tr_geo038_draw_layout(indices, groups, vertex_count):
    """Model 0.18.0 TerrainBarn's 36B vertex duplication / remap loop.

    A reused source vertex is duplicated when its distance from the previous
    mapped index exceeds (vertices newly added in this record + 256). Check the
    resulting per-record span, not merely global source IDs or index count.
    """
    mapping, previous, output_count, max_span = [-1] * vertex_count, -1, 0, 0
    for group in groups:
        for row in group.get('records') or []:
            added, lo, hi = 0, float('inf'), -1
            for index in indices[int(row['idx_start']):int(row['idx_start']) + int(row['idx_count'])]:
                mapped = mapping[index]
                if mapped == -1 or abs(mapped - previous) > added + 256:
                    mapped = output_count
                    output_count += 1
                    added += 1
                    mapping[index] = mapped
                lo, hi, previous = min(lo, mapped), max(hi, mapped), mapped
            span = hi - lo + 1 if hi >= 0 else 0
            _tr_require(span <= 65535 and int(row['idx_count']) <= 65535,
                        '0x38 加载器重排后的绘制记录超出 uint16 局部跨度/索引数限制')
            max_span = max(max_span, span)
    return {'loader_repacked_vertices': output_count, 'max_draw_vertex_span': max_span}


def _tr_validate_geo038_render_tables(geo0):
    """Weight/pass, AABB and uint16 draw validation for rebuilt 0x38 only."""
    _tr_require(geo0 and geo0.get('layout') == 'geo0-0x38', '目标不是 0x38 GEO0')
    raw = _join_verts(geo0.get('v') or {}, 36)
    _tr_require(raw is not None, '目标 0x38 GEO0 顶点字段不完整')
    indices = geo0.get('indices')
    if indices is None:
        indices = meshopt_decode_index_buffer(int(geo0.get('idx_count', 0)),
                                             _b64d(geo0.get('ib_b64', '')), 4)
    count = _tr_check_surface(raw, 36, indices, '目标 0x38 GEO0')
    _tr_require(len(indices) == int(geo0.get('idx_count', len(indices))), '0x38 索引计数不符')
    groups = geo0.get('draw_groups') or []
    _tr_require(len({int(g['material_id']) for g in groups}) == len(groups), '0x38 材质组重复')
    cache, expected, actual = {}, {}, {}
    counts = {'vertices': count, 'draw_groups': len(groups), 'draw_records': 0,
              'terrain_triangles': 0, 'cloud_triangles': 0, 'material_pass_triangles': 0}
    intervals = []
    for group in groups:
        material = int(group['material_id'])
        _tr_require(0 <= material < 255, '0x38 材质 ID 无效')
        counter = actual.setdefault(material, _tr_Counter())
        for row in group.get('records') or []:
            start, length = int(row['idx_start']), int(row['idx_count'])
            _tr_require(start >= 0 and start % 3 == 0 and 0 < length <= 1500 and length % 3 == 0 and
                        start + length <= len(indices), '0x38 绘制记录索引范围超限')
            intervals.append((start, start + length))
            # The exact loader remap/span check runs below after all passes.
            for axis in range(3):
                lo, hi = float(row['min'][axis]), float(row['max'][axis])
                _tr_require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
                            '0x38 绘制记录包围盒无效')
                _tr_require(all(lo <= struct.unpack_from('<3f', raw, index * 36)[axis] <= hi
                                for index in set(indices[start:start + length])),
                            '0x38 绘制记录包围盒未覆盖顶点')
            for at in range(start, start + length, 3):
                triangle = indices[at:at + 3]
                for index in triangle:
                    if index not in cache:
                        cache[index] = _tr_geo_material_weights(raw, index)
                active = set().union(*(cache[index] for index in triangle))
                key = _tr_oriented_key(triangle)
                counter[key] += 1
                if material == 0:
                    _tr_require(active and 80 not in active, '0x38 深度表混入云/零权重地形')
                    counts['terrain_triangles'] += 1
                    for mat in active:
                        expected.setdefault(mat, _tr_Counter())[key] += 1
                elif material == 80:
                    _tr_require(active == {80}, '0x38 云表材质无效')
                    counts['cloud_triangles'] += 1
                else:
                    _tr_require(material in active, '0x38 材质 pass 包含零权重三角形')
                    counts['material_pass_triangles'] += 1
            counts['draw_records'] += 1
    _tr_require({mat: counter for mat, counter in actual.items() if mat not in (0, 80)} == expected,
                '0x38 材质 pass 的有向面/重复次数与深度表权重不符')
    cursor = 0
    for start, end in sorted(intervals):
        _tr_require(start == cursor, '0x38 索引缓冲有遗漏/重叠的绘制范围')
        cursor = end
    _tr_require(cursor == len(indices), '0x38 存在未引用索引')
    counts.update(_tr_geo038_draw_layout(indices, groups, count))
    counts.update({'all_triangles_material_covered': True, 'uint16_draw_limits_verified': True,
                   'in_game_runtime_tested': False})
    return counts


def _tr_make_modern_geo0(surfaces, version):
    """Rebuild real 0x39+ GEO0, never substitute terrain with LOD0 skirts.

    Meshlets use <=252 local vertices / <=252 oriented triangles. Subchunks
    describe every material whose nonzero vertex weight touches a triangle;
    the game filters their inclusive ranges and requires exact vertex/face
    counts. Cone byte 127 disables backface-cone culling (sub_2366320).
    """
    _tr_require(version >= 0x39, '现代 GEO0 重建需要 0x39 或更新目标')
    prepared = []
    for raw, stride, indices, label in surfaces:
        _tr_require(stride == 36, '地形 GEO0 顶点必须为 36B')
        _tr_check_surface(raw, 36, indices, '源 ' + label)
        material_cache = {i: _tr_geo_material_weights(raw, i) for i in set(indices)}
        categories = {'terrain': [], 'cloud': []}
        for at in range(0, len(indices), 3):
            triangle = indices[at:at + 3]
            materials = set().union(*(material_cache[i] for i in triangle))
            _tr_require(materials and all(0 < mat < 255 for mat in materials),
                        '源地形有零权重/无效材质三角形，不能声明目标可绘制')
            _tr_require(80 not in materials or materials == {80},
                        '混合体积云/普通地形材质尚不能转为目标云分块；拒绝静默丢失材质')
            category = 'cloud' if materials == {80} else 'terrain'
            categories[category].append((triangle, materials))
        for category, triangles in categories.items():
            if triangles:
                prepared.append((category, raw, material_cache, triangles))
    if not prepared:
        return _tr_empty_geo(version)

    vertices, ib, subs = bytearray(), bytearray(), []
    chunks, cloud_chunks = [], []
    for category in ('terrain', 'cloud'):
        destination = chunks if category == 'terrain' else cloud_chunks
        for group_category, raw, material_cache, source_triangles in prepared:
            if group_category != category:
                continue
            mapping, local, source_vertices, triangle_materials = {}, [], [], []

            def finish():
                if not local:
                    return
                vertex_start, index_start, area_start = len(vertices) // 36, len(ib), len(subs)
                points = []
                for index in source_vertices:
                    at = index * 36
                    vertices.extend(raw[at:at + 36])
                    points.append(struct.unpack_from('<3f', raw, at))
                for material in sorted(set().union(*triangle_materials)):
                    selected = [at for at, materials in enumerate(triangle_materials)
                                if material in materials]
                    referenced = set(index for at in selected for index in local[at * 3:at * 3 + 3])
                    subs.append({'material_id': material, 'triangle_count': len(selected),
                                 'vtx_count': len(referenced), 'triangle_start': selected[0],
                                 'triangle_end': selected[-1], 'vtx_start': min(referenced),
                                 'vtx_end': max(referenced), 'unk': 0})
                ib.extend(local)
                destination.append({'idx_start': index_start, 'vtx_start': vertex_start,
                                    'area_start': area_start, 'idx_count': len(local),
                                    'vtx_count': len(source_vertices),
                                    'area_count': len(subs) - area_start,
                                    'min': [min(p[axis] for p in points) - 0.1 for axis in range(3)],
                                    'max': [max(p[axis] for p in points) + 0.1 for axis in range(3)],
                                    'unk': [0, 0, 0, 0x7F000000]})

            for triangle, materials in source_triangles:
                new_count = len(set(triangle).difference(mapping))
                if local and (len(local) + 3 > 756 or len(mapping) + new_count > 252):
                    finish()
                    mapping, local, source_vertices, triangle_materials = {}, [], [], []
                for index in triangle:
                    if index not in mapping:
                        mapping[index] = len(mapping)
                        source_vertices.append(index)
                    local.append(mapping[index])
                triangle_materials.append(materials)
            finish()
    result = {'vert_count': len(vertices) // 36, 'v': _split_verts(bytes(vertices), len(vertices) // 36, 36),
              'vb_comp_b64': '', 'vb_hash': '', 'ib_b64': _b64e(bytes(ib)),
              'chunks': chunks, 'cloud_chunks': cloud_chunks, 'subchunks': subs}
    if version >= 0x3D:
        # 0.35.1 LoadLod reads optional wayfinderRegionCount after the subchunks.
        # A converted old map has no such navigation cells, so emit explicit 0.
        result['trailing_b64'] = _b64e(struct.pack('<I', 0))
    _tr_validate_geo0_render_tables(result)
    return result


# ---- experimental cloud migration (isolated candidate, 2026-11-08) ------------
_TR_CLOUD_RAW_LIMIT = 128 * 1024 * 1024


def _tr_cloud_lz4_unpack(cloud, key, size, slot):
    """Cloud blobs are raw LZ4 blocks, not the old generic LZ77 codec."""
    _tr_require(0 <= size <= _TR_CLOUD_RAW_LIMIT, '云采样解压大小超出安全上限')
    packed = bytes(cloud.get(key) or [])
    sizes = cloud.get('v3_sizes') or []
    _tr_require(len(sizes) > slot and int(sizes[slot]) == len(packed),
                '云 %s 压缩长度字段与实际数据不符' % key)
    if not size and not packed:
        return b''
    try:
        raw = lz4.block.decompress(packed, uncompressed_size=max(size, 1))
    except Exception as exc:
        raise MeshesFormatError('云 %s LZ4 解压失败：%s' % (key, exc))
    _tr_require(len(raw) == size, '云 %s 解压长度不符' % key)
    return raw


def _tr_cloud_unpack(cloud, version):
    origin, dim = cloud.get('bbox_origin'), cloud.get('dim')
    valid_int = lambda x: isinstance(x, int) and not isinstance(x, bool)
    _tr_require(isinstance(origin, list) and len(origin) == 3 and
                all(valid_int(x) and -0x80000000 <= x <= 0x7FFFFFFF for x in origin),
                '云网格原点无效')
    _tr_require(isinstance(dim, list) and len(dim) == 3 and
                all(valid_int(x) and 0 < x <= 32768 for x in dim), '云网格维度无效')
    cells = math.prod(dim)
    _tr_require(cells <= 64_000_000, '云粗网格超过安全上限')
    coords = cloud.get('index_entries') or []
    count = int(cloud.get('count1', -1))
    _tr_require(0 <= count <= cells and count == len(coords), '云 brick 数量不符')
    _tr_require(all(isinstance(p, (list, tuple)) and len(p) == 3 and
                    all(valid_int(x) and 0 <= x < dim[a] and x <= 32767
                        for a, x in enumerate(p)) for p in coords), '云 brick 坐标越界')
    _tr_require(len({tuple(p) for p in coords}) == count, '云 brick 坐标重复')
    b = cloud.get('bpp_dist_amb01')
    a = (cloud.get('v6') or [None])[0] if version <= 0x36 else cloud.get('bpp_amb2')
    _tr_require(all(valid_int(x) and 1 <= x <= 32 for x in (b, a)), '云采样分辨率无效')
    scale = cloud.get('voxel_scale')
    _tr_require(isinstance(scale, (int, float)) and math.isfinite(scale), '云材质浮点字段无效')
    lengths = (count * b ** 3, count * a ** 3 * 8, count * a ** 3)
    _tr_require(sum(lengths) + cells * 2 <= _TR_CLOUD_RAW_LIMIT,
                '云采样总解压大小超过 128 MiB 安全上限')
    raw = {key: _tr_cloud_lz4_unpack(cloud, key, size, i)
           for i, (key, size) in enumerate(zip(('packed_dist', 'packed_amb0', 'packed_amb1'), lengths))}
    if version <= 0x36:
        _tr_require(int(cloud.get('sparse_total', cells)) == cells, '旧云粗网格长度不符')
        coarse, seen = bytearray([255]) * cells, set()
        for row in cloud.get('sparse_non_empty') or []:
            _tr_require(isinstance(row, (list, tuple)) and len(row) == 4 and
                        all(valid_int(x) for x in row), '旧云粗网格条目无效')
            x, y, z, value = row
            _tr_require(0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2] and
                        0 <= value <= 255, '旧云粗网格条目越界')
            at = x + dim[0] * (y + dim[1] * z)
            _tr_require(at not in seen, '旧云粗网格坐标重复')
            seen.add(at)
            coarse[at] = value
        coarse = bytes(coarse)
    elif version < 0x3C:
        mask = cloud.get('mask') or []
        _tr_require(len(mask) == cells and all(valid_int(x) and 0 <= x <= 65535 for x in mask),
                    '云 half 粗网格长度/值无效')
        coarse = struct.pack('<%dH' % cells, *mask)
    else:
        coarse = _tr_cloud_lz4_unpack(cloud, 'packed_extra', cells * 2, 3)
    if version >= 0x37:
        _tr_require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<e', coarse)),
                    '云粗距离含 NaN/Infinity，拒绝跨版转换')
    raw['coarse'] = coarse
    return raw, b, a


def _tr_cloud_cpu_distance(value, version):
    if version < 0x3C:
        return value * (10.0 / 255.0) - 6.0
    signed = value if value < 128 else value - 256
    return 5.0 if value == 127 else signed * (5.0 / 128.0)


def _tr_cloud_gpu_distance(value, version):
    if version < 0x3C:
        return value * (10.0 / 255.0) - 6.0
    signed = value if value < 128 else value - 256
    return max(-1.0, signed / 127.0) * 5.0  # VK_FORMAT_R8_SNORM


def _tr_cloud_convert_dist(raw, source_version, target_version):
    if (source_version < 0x3C) == (target_version < 0x3C):
        return raw, {'converted': False, 'samples': len(raw), 'saturated_samples': 0,
                     'max_abs_cpu_error': 0.0, 'max_abs_gpu_error': 0.0}
    table = []
    for value in range(256):
        distance = _tr_cloud_cpu_distance(value, source_version)
        if target_version >= 0x3C:
            # Match the observed CPU /128 model, including native -128 and
            # +127 endpoints. The GPU uses /127; report that drift separately.
            encoded = min(127, max(-128, round(distance * (128.0 / 5.0)))) & 255
        else:
            encoded = min(255, max(0, round((distance + 6.0) * (255.0 / 10.0))))
        table.append(encoded)
    stats = {'converted': True, 'samples': len(raw), 'quantized_samples': 0,
             'saturated_samples': 0, 'max_abs_cpu_error': 0.0, 'max_abs_gpu_error': 0.0,
             'max_unsaturated_cpu_error': 0.0, 'max_unsaturated_gpu_error': 0.0,
             'opposite_sign_samples': 0, 'quantization_reference': 'observed CPU distance',
             'gpu_model': 'R8_SNORM /127; CPU /128 with 0x7F=+5',
             'collision_equivalence_guaranteed': False}
    low, high = (-5.0, 5.0) if target_version >= 0x3C else (-6.0, 4.0)
    for value, count in _tr_Counter(raw).items():
        before = _tr_cloud_cpu_distance(value, source_version)
        after = _tr_cloud_cpu_distance(table[value], target_version)
        cpu_error = abs(before - after)
        gpu_error = abs(_tr_cloud_gpu_distance(value, source_version) -
                        _tr_cloud_gpu_distance(table[value], target_version))
        saturated = not low <= before <= high
        stats['quantized_samples'] += count if cpu_error > 1e-12 else 0
        stats['saturated_samples'] += count if saturated else 0
        stats['opposite_sign_samples'] += count if before * after < 0 else 0
        stats['max_abs_cpu_error'] = max(stats['max_abs_cpu_error'], cpu_error)
        stats['max_abs_gpu_error'] = max(stats['max_abs_gpu_error'], gpu_error)
        if not saturated:
            stats['max_unsaturated_cpu_error'] = max(stats['max_unsaturated_cpu_error'], cpu_error)
            stats['max_unsaturated_gpu_error'] = max(stats['max_unsaturated_gpu_error'], gpu_error)
    return raw.translate(bytes(table)), stats


def _tr_cloud_convert_coarse(raw, source_version, target_version, dim):
    if (source_version <= 0x36) == (target_version <= 0x36):
        return raw, {'converted': False, 'max_abs_shader_distance_error': 0.0}
    stats = {'converted': True, 'reference': 'legacy byte/255*110-10; modern half*0.8',
             'max_abs_shader_distance_error': 0.0, 'saturated_cells': 0,
             'cpu_fallback_equivalence_guaranteed': False}
    result = bytearray()
    if source_version <= 0x36:
        for value in raw:
            distance = value * (110.0 / 255.0) - 10.0
            encoded = struct.pack('<e', distance / 0.8)
            after = struct.unpack('<e', encoded)[0] * 0.8
            stats['max_abs_shader_distance_error'] = max(stats['max_abs_shader_distance_error'],
                                                         abs(distance - after))
            result.extend(encoded)
    else:
        for (value,) in struct.iter_unpack('<e', raw):
            distance = value * 0.8
            encoded = min(255, max(0, round((distance + 10.0) * (255.0 / 110.0))))
            after = encoded * (110.0 / 255.0) - 10.0
            stats['max_abs_shader_distance_error'] = max(stats['max_abs_shader_distance_error'],
                                                         abs(distance - after))
            stats['saturated_cells'] += not -10.0 <= distance <= 100.0
            result.append(encoded)
    return bytes(result), stats


def _tr_convert_cloud_volume(cloud, source_version, target_version):
    # 0x37 has a separate meshopt proxy metadata stream whose construction is
    # not yet proved. Do not write guessed metadata or declare an empty proxy
    # equivalent to the source render mesh.
    if target_version == 0x37:
        return None, {'converted': False, 'unsupported_reason':
                      '0x37 独立云 proxy 元数据生成规则尚未确认；不写入猜测结构'}
    raw, b, a = _tr_cloud_unpack(cloud, source_version)
    distance, distance_stats = _tr_cloud_convert_dist(raw['packed_dist'], source_version, target_version)
    coarse, coarse_stats = _tr_cloud_convert_coarse(raw['coarse'], source_version, target_version, cloud['dim'])
    result = {'present': True, 'version': target_version, 'bbox_origin': list(cloud['bbox_origin']),
              'dim': list(cloud['dim']), 'count1': cloud['count1'],
              'index_entries': copy.deepcopy(cloud['index_entries']),
              'voxel_scale': cloud['voxel_scale'], 'bpp_dist_amb01': b}
    result['packed_dist'] = (list(cloud['packed_dist']) if not distance_stats['converted'] else
                             list(_lz4_compress(distance)))
    # Retain native baked channels, including point exponent/mantissa. Legacy
    # game shaders can interpret these differently: preserving samples is not
    # a proof of cross-game radiometric equivalence.
    for key in ('packed_amb0', 'packed_amb1'):
        result[key] = list(cloud[key])
    if target_version <= 0x36:
        gx, gy, gz = result['dim']
        result['sparse_total'] = gx * gy * gz
        result['sparse_non_empty'] = [[i % gx, (i // gx) % gy, i // (gx * gy), value]
                                     for i, value in enumerate(coarse) if value != 255]
        result['v6'] = (copy.deepcopy(cloud['v6']) if source_version <= 0x36 else [a, 0, 0])
        result['v6_v'] = (copy.deepcopy(cloud.get('v6_v') or {}) if source_version <= 0x36 else
                          {key: [] for key in ('px', 'py', 'pz', 'normal_u', 'normal_v')})
        result['v6_indices'] = (list(cloud.get('v6_indices') or []) if source_version <= 0x36 else [])
        if target_version == 0x36:
            result['trail'] = 0
    else:
        result['bpp_amb2'] = a
        if target_version < 0x3C:
            result['mask'] = list(struct.unpack('<%dH' % math.prod(result['dim']), coarse))
            result['mask_total'] = math.prod(result['dim'])
        else:
            result['packed_extra'] = (list(cloud['packed_extra']) if source_version >= 0x3C else
                                       list(_lz4_compress(coarse)))
    keys = ['packed_dist', 'packed_amb0', 'packed_amb1'] + (['packed_extra'] if target_version >= 0x3C else [])
    result['v3_sizes'] = [len(result[key]) for key in keys]
    stats = {'converted': True, 'bricks': cloud['count1'], 'distance_resolution': b,
             'lighting_resolution': a, 'bin_world_units': 16, 'material_float_preserved': True,
             'origin_dimensions_brick_order_preserved': True, 'distance': distance_stats,
             'coarse': coarse_stats, 'lighting_blobs_byte_exact': True,
             'lighting_shader_equivalence_guaranteed': False,
             'lighting_policy': 'preserve source packed baked channels; sample target encoding for new mesh',
             'source_proxy_metadata_archived': source_version == 0x37,
             'source_trail_archived': bool(cloud.get('trail', 0)),
             'decompressed_bytes': {key: len(value) for key, value in raw.items()},
             'in_game_runtime_tested': False}
    return result, stats


def _tr_cloud_proxy_surface(cloud, version):
    if not cloud.get('present') or version >= 0x38:
        return None
    if version <= 0x36:
        a, count, index_count = cloud.get('v6') or [0, 0, 0]
        v = cloud.get('v6_v') or {}
        _tr_require(all(len(v.get(key) or []) == count
                        for key in ('px', 'py', 'pz', 'normal_u', 'normal_v')),
                    '旧云代理顶点字段长度不符')
        raw = b''.join(struct.pack('<3fI', v['px'][i], v['py'][i], v['pz'][i],
                                  v['normal_u'][i] | (v['normal_v'][i] << 16)) for i in range(count))
        indices = list(cloud.get('v6_indices') or [])
        stride = 16
    else:
        proxy = cloud.get('proxy') or {}
        _tr_require(not proxy.get('vb_error') and not proxy.get('ib_error'), '0x37 云代理未完整解码')
        count, index_count = int(proxy.get('vert_count', 0)), int(proxy.get('index_count', 0))
        raw, stride = _join_verts(proxy.get('v') or {}, 20), 20
        _tr_require(raw is not None, '0x37 云代理顶点字段不完整')
        indices = meshopt_decode_index_buffer(index_count, _b64d(proxy.get('ib_b64', '')), 2) if index_count else []
    _tr_require(raw is not None and len(raw) == count * stride and len(indices) == index_count,
                '云代理顶点/索引计数不符')
    _tr_require(len(indices) % 3 == 0 and all(isinstance(i, int) and 0 <= i < count for i in indices),
                '云代理索引截断/越界')
    _tr_require(all(math.isfinite(x) for i in range(count)
                    for x in struct.unpack_from('<3f', raw, i * stride)), '云代理含非有限坐标')
    return (raw, stride, indices, 'cloud_proxy') if indices else None


def _tr_cloud_bake_proxy(surface, target_cloud, target_version):
    """Sample encoded lighting with the target loader's A+1 border rules.

    CloudShMesh interpolates U8 baked attributes. Never use the unrelated
    octahedral-normal helpers: proxy normal XYZ are signed byte lanes.
    No world scaling, surface shrink or fabricated terrain/collision is added.
    """
    raw, stride, indices, label = surface
    samples, b, a = _tr_cloud_unpack(target_cloud, target_version)
    coords = {tuple(p): i for i, p in enumerate(target_cloud['index_entries'])}
    origin = target_cloud['bbox_origin']
    count, baked = len(raw) // stride, bytearray(len(raw) // stride * 36)
    stats = {'source_vertices': count, 'source_triangles': len(indices) // 3,
             'inactive_bin_vertices': 0, 'zero_normal_vertices': 0,
             'light_sampling': 'target encoded trilinear A+1 atlas, adjacent brick or clamp',
             'light2': 'source packed normal XYZ biased to unsigned bytes; w=0',
             'visual_equivalence_guaranteed': False, 'in_game_runtime_tested': False}
    default = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 128 / 255.0, 0.0]
    for i in range(count):
        src, dst = i * stride, i * 36
        position = struct.unpack_from('<3f', raw, src)
        grid = [position[k] / 16.0 - origin[k] for k in range(3)]
        base = tuple(math.floor(x) for x in grid)
        if base not in coords:
            light = default
            stats['inactive_bin_vertices'] += 1
        else:
            local = [(grid[k] - base[k]) * a for k in range(3)]
            low = [math.floor(x) for x in local]
            frac = [local[k] - low[k] for k in range(3)]
            light = [0.0] * 8
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        corner = (dx, dy, dz)
                        weight = math.prod(frac[k] if corner[k] else 1.0 - frac[k] for k in range(3))
                        if not weight:
                            continue
                        voxel = [low[k] + corner[k] for k in range(3)]
                        neighbor = tuple(base[k] + (voxel[k] == a) for k in range(3))
                        if neighbor in coords:
                            ordinal = coords[neighbor]
                            voxel = [0 if x == a else x for x in voxel]
                        else:
                            ordinal = coords[base]
                            voxel = [min(x, a - 1) for x in voxel]
                        at = 8 * (ordinal * a ** 3 + voxel[0] + a * (voxel[1] + a * voxel[2]))
                        for channel in range(8):
                            light[channel] += samples['packed_amb0'][at + channel] * weight / 255.0
        encoded = [min(255, max(0, round(x * 255.0))) for x in light]
        _tr_require(encoded[3] > 0, '云代理采样得到零光照分母，拒绝生成无效 baked light0')
        normal = list(struct.unpack_from('<3b', raw, src + 12))
        _tr_require(any(normal), '云代理含零法线，拒绝注入会 normalize(0) 的 GEO0 云顶点')
        baked[dst:dst + 15] = raw[src:src + 15]
        baked[dst + 15] = 0
        baked[dst + 16:dst + 24] = b'\x50\x00\x00\x00\xff\x00\x00\x00'
        baked[dst + 24:dst + 32] = bytes(encoded)
        baked[dst + 32:dst + 36] = bytes([x + 128 for x in normal] + [0])
    stats['source_proxy_extra_channels_archived'] = stride == 20
    return (bytes(baked), 36, indices, 'cloud'), stats


def _tr_cloud_make_legacy_proxy(surfaces, cloud):
    vertices, indices = bytearray(), []
    for raw, stride, source_indices, label in surfaces:
        mapping = {}
        for index in source_indices:
            if index not in mapping:
                _tr_require(len(vertices) // 16 < 65536, '目标旧云独立代理超过 uint16 顶点上限')
                mapping[index] = len(vertices) // 16
                at = index * stride
                vertices.extend(raw[at:at + 15] + b'\x00')
            indices.append(mapping[index])
    count = len(vertices) // 16
    v = {key: [] for key in ('px', 'py', 'pz', 'normal_u', 'normal_v')}
    for x, y, z, normal in struct.iter_unpack('<3fI', vertices):
        for key, value in zip(v, (x, y, z, normal & 65535, normal >> 16)):
            v[key].append(value)
    cloud['v6'] = [cloud['v6'][0], count, len(indices)]
    cloud['v6_v'], cloud['v6_indices'] = v, indices
    return {'vertices': count, 'triangles': len(indices) // 3,
            'source_baked_vertex_attributes_archived': any(s[1] >= 20 for s in surfaces),
            'in_game_runtime_tested': False}


def _tr_geometry_surfaces(decoded):
    surfaces = [(_tr_terrain_raw(p), 36, _tr_terrain_indices(p), 'terrain')
                for p in decoded.get('terrain_patches') or []]
    return surfaces + _tr_geo_surfaces(decoded.get('geo0'), decoded['version'])


def _tr_cloud_split_surfaces(surfaces):
    """Separate material-80 render faces, never the unrelated skirt groups."""
    ordinary, clouds = [], []
    for raw, stride, indices, label in surfaces:
        _tr_check_surface(raw, stride, indices, label, check_positions=False)
        cache = {i: _tr_geo_material_weights(raw, i) for i in set(indices)}
        other_indices, cloud_indices = [], []
        for at in range(0, len(indices), 3):
            triangle = indices[at:at + 3]
            materials = set().union(*(cache[i] for i in triangle))
            _tr_require(80 not in materials or materials == {80},
                        '混合体积云/普通地形材质不能安全分离；拒绝丢失材质')
            (cloud_indices if materials == {80} else other_indices).extend(triangle)
        if other_indices:
            ordinary.append((raw, stride, other_indices, 'terrain'))
        if cloud_indices:
            clouds.append((raw, stride, cloud_indices, 'cloud'))
    return ordinary, clouds


def _tr_cloud_surface_digest(surfaces):
    """Cloud positions + packed normal XYZ, with oriented face multiplicity.

    A 16B/20B independent proxy cannot preserve GEO0 material/lighting lanes.
    Those bytes are checked separately when retained, or explicitly archived
    on a downgrade. Do not label this narrower digest full vertex equivalence.
    """
    hashes = []
    for raw, stride, indices, label in surfaces:
        _tr_require(stride in (16, 20, 36) and len(raw) % stride == 0,
                    '云几何顶点步长无效')
        count = len(raw) // stride
        _tr_require(len(indices) % 3 == 0 and all(isinstance(i, int) and
                    not isinstance(i, bool) and 0 <= i < count for i in indices),
                    '云几何索引截断/越界')
        tokens = {}
        for index in set(indices):
            at = index * stride
            _tr_require(all(math.isfinite(x) for x in struct.unpack_from('<3f', raw, at)),
                        '云几何含非有限坐标')
            tokens[index] = raw[at:at + 15]
        for at in range(0, len(indices), 3):
            a, b, c = (tokens[i] for i in indices[at:at + 3])
            hashes.append(hashlib.sha256(min(a + b + c, b + c + a, c + a + b)).digest())
    hashes.sort()
    return {'triangles': len(hashes), 'sha256': hashlib.sha256(b''.join(hashes)).hexdigest(),
            'vertex_fields': 'float32 position XYZ + packed normal XYZ; winding and multiplicity'}


def _tr_cloud_render_digest(decoded):
    ordinary, surfaces = _tr_cloud_split_surfaces(_tr_geometry_surfaces(decoded))
    proxy = _tr_cloud_proxy_surface(decoded.get('cloud') or {}, decoded['version'])
    if proxy:
        surfaces.append(proxy)
    return _tr_cloud_surface_digest(surfaces)


def _tr_cloud_family(version):
    if version <= 0x36:
        return 'legacy'
    if version == 0x37:
        return 'proxy'
    return 'maskless' if version >= 0x3C else 'mask16'


def _tr_cloud_is_portable(cloud, source_version, target_version):
    if not cloud or not cloud.get('present'):
        return True
    if _tr_cloud_family(source_version) != _tr_cloud_family(target_version):
        return False
    if (source_version >= 0x36) != (target_version >= 0x36) and cloud.get('trail', 0):
        return False
    return True


def _tr_source_issues(decoded):
    issues = []
    if decoded.get('lod0_undecoded'):
        issues.append('LOD0 未解码')
    if decoded.get('_remaining'):
        issues.append('LOD0 有未解释尾部字节')
    if (decoded.get('container') or {}).get('extra_sections'):
        issues.append('存在未知容器分段')
    geo = decoded.get('geo0') or {}
    if geo.get('opaque'):
        issues.append('GEO0 不透明/未知布局')
    if geo.get('trailing_b64'):
        trailing = _b64d(geo['trailing_b64'])
        # 0x3D's explicit empty wayfinderRegionCount is known metadata, not an
        # unknown tail. Nonempty navigation regions remain archive-only.
        if not (int(decoded.get('version', 0)) >= 0x3D and trailing == struct.pack('<I', 0)):
            issues.append('GEO0 有未解释尾部字节')
    return issues


def _tr_quiet_decode(data):
    with _tr_contextlib.redirect_stdout(_tr_io.StringIO()):
        return decode_meshes_bytes(data)


def _tr_normalize_input(decoded):
    _tr_require(isinstance(decoded, dict), '输入 JSON 根节点必须为对象')
    _tr_require(decoded.get('format') in (None, 'sky-meshes-v2', 'sky-meshes-v3'),
                '仅支持本脚本解码的 JSON，不支持其他解码器的不同 schema')
    if 'version' in decoded:
        version = parse_meshes_version(decoded['version'])
    elif decoded.get('version_hex'):
        version = parse_meshes_version(decoded['version_hex'])
    else:
        # Historical output from this same file was single-version (0x36).
        version = 0x36
    if decoded.get('version_hex'):
        _tr_require(parse_meshes_version(decoded['version_hex']) == version,
                    'version 与 version_hex 不一致；不要手动改版本，请用功能 3')
    out = copy.deepcopy(decoded)
    if out.get('container') is None:
        out['container'] = {}
    out['version'] = version
    out['version_hex'] = '0x%02X' % version
    out.setdefault('format', 'sky-meshes-v2')
    out.setdefault('container', {})
    out.setdefault('geo0', None)
    out.setdefault('metr', None)
    out.setdefault('generic_meshes', [])
    out.setdefault('terrain_patches', [])
    out.setdefault('cloud', {'present': False, 'version': version})
    out.setdefault('cloud_mesh', {'groups': [], 'mesh_16': None, 'version': version})
    cloud = out.get('cloud') or {}
    # Names emitted by the old single-version JSON writer.
    if 'voxel_scale' not in cloud and 'float_v5' in cloud:
        cloud['voxel_scale'] = cloud['float_v5']
    if 'bpp_dist_amb01' not in cloud and 'v5_single' in cloud:
        cloud['bpp_dist_amb01'] = cloud['v5_single']
    geo0 = out.get('geo0') or {}
    if geo0.get('opaque') and version == 0x38 and geo0.get('raw_b64'):
        candidate = decode_geo0(_b64d(geo0['raw_b64']), version)
        if candidate and not candidate.get('opaque'):
            out['geo0'] = candidate
    return out


def _tr_bounds(decoded):
    old = decoded.get('bounds') or {}
    lo, hi = [float('inf')] * 3, [float('-inf')] * 3
    if (len(old.get('min', [])) == 3 and len(old.get('max', [])) == 3 and
            all(math.isfinite(x) for x in old['min'] + old['max']) and
            all(a <= b for a, b in zip(old['min'], old['max']))):
        lo, hi = list(old['min']), list(old['max'])
    objects = [p.get('v') or {} for p in decoded.get('terrain_patches') or []]
    objects.append((decoded.get('geo0') or {}).get('v') or {})
    cm, cloud = decoded.get('cloud_mesh') or {}, decoded.get('cloud') or {}
    objects.extend(g.get('v') or {} for g in cm.get('groups') or [])
    objects.append((cm.get('mesh_16') or {}).get('v') or {})
    objects.append(cloud.get('v6_v') or {})
    objects.append((cloud.get('proxy') or {}).get('v') or {})
    for v in objects:
        for point in zip(v.get('px') or [], v.get('py') or [], v.get('pz') or []):
            _tr_require(all(math.isfinite(x) for x in point), '包围盒计算遇到非有限坐标')
            for axis, value in enumerate(point):
                lo[axis] = min(lo[axis], value)
                hi[axis] = max(hi[axis], value)
    if lo[0] == float('inf'):
        return {'min': [0.0] * 3, 'max': [0.0] * 3}
    return {'min': lo, 'max': hi}


def _tr_empty_geo(version):
    if version == 0x38:
        return {'layout': 'geo0-0x38', 'stride': 36, 'index_stride': 4,
                'vert_count': 0, 'idx_count': 0, 'v': _split_verts(b'', 0, 36),
                'vb_comp_b64': '', 'vb_hash': _md5(b''), 'indices': [],
                'ib_b64': '', 'ib_hash': _md5(b''), 'draw_groups': []}
    return {'empty_header20': True, 'vert_count': 0, 'v': _split_verts(b'', 0, 36),
            'chunks': [], 'cloud_chunks': [], 'subchunks': [], 'ib_b64': '',
            'vb_comp_b64': '', 'vb_hash': _md5(b'')}


def _tr_surface_digest(decoded, alpha_target_version=None, exclude_cloud_material=False,
                       extra_surfaces=None):
    """Digest rendered oriented triangles using 36B vertices + effective alpha.

    alpha_target_version models only the declared/checked downgrade quantization
    when the target legacy alpha byte cannot express a modern value exactly.

    Independent of chunk packing, uint16/meshopt encoding, and triangle rotation.
    Multiplicity and winding are preserved. Old skirt-only XYZ bytes at 36..38
    are not part of this geometric digest; their loss is reported separately.
    """
    version = parse_meshes_version(decoded['version'])
    surfaces = []
    for i, patch in enumerate(decoded.get('terrain_patches') or []):
        surfaces.append((_tr_terrain_raw(patch), 36, _tr_terrain_indices(patch), 'terrain'))
    surfaces.extend(_tr_geo_surfaces(decoded.get('geo0'), version))
    if exclude_cloud_material:
        surfaces, _cloud_surfaces = _tr_cloud_split_surfaces(surfaces)
    surfaces.extend(extra_surfaces or [])
    for i, group in enumerate((decoded.get('cloud_mesh') or {}).get('groups') or []):
        surfaces.append(_tr_skirt_surface(group, version, 'skirt_%d' % i))
    hashes, checked = [], set()
    for raw, stride, indices, label in surfaces:
        key = (id(raw), stride)
        _tr_check_surface(raw, stride, indices, label, check_positions=key not in checked)
        checked.add(key)
        tokens = {}
        skirt = label.startswith('skirt')
        for index in set(indices):
            start = index * stride
            token = bytearray(raw[start:start + 36])
            if skirt:
                value = raw[start + 39] if stride == 40 else raw[start + 15]
                if alpha_target_version is not None:
                    value = _tr_skirt_alpha_to_version(value, version, alpha_target_version)
                    alpha = _tr_skirt_runtime_alpha(value, alpha_target_version)
                else:
                    alpha = _tr_skirt_runtime_alpha(value, version)
            else:
                alpha = 127
            token[15] = 0  # normalize unused legacy normal.w / new alpha storage
            token.append(alpha)
            tokens[index] = bytes(token)
        for at in range(0, len(indices), 3):
            a, b, c = (tokens[x] for x in indices[at:at + 3])
            hashes.append(hashlib.sha256(min(a + b + c, b + c + a, c + a + b)).digest())
    hashes.sort()
    return {'triangles': len(hashes), 'sha256': hashlib.sha256(b''.join(hashes)).hexdigest()}


def _tr_archive_pack(original, encoded):
    # Keep the EXACT input JSON, not just a canonical re-decode: annotations,
    # opaque fields, and source compression caches remain recoverable.
    text = json.dumps(original, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    version = parse_meshes_version(original.get('version', original.get('version_hex', 0x36)))
    _tr_require(len(text) <= 512 * 1024 * 1024, '源 JSON 超过归档的 512 MiB 上限')
    return {'format': 'sky-meshes-translation-archive-v1', 'source_version': version,
            'source_version_hex': '0x%02X' % version,
            'json_codec': 'zlib+base64', 'json_bytes': len(text),
            'json_sha256': hashlib.sha256(text).hexdigest(),
            'json_b64': _b64e(_tr_zlib.compress(text, 6)),
            'meshes_bytes': len(encoded), 'meshes_sha256': hashlib.sha256(encoded).hexdigest()}


def _tr_archive_unpack(archive):
    _tr_require(isinstance(archive, dict) and
                archive.get('format') == 'sky-meshes-translation-archive-v1', '转译归档格式无效')
    size = archive.get('json_bytes')
    _tr_require(isinstance(size, int) and 0 < size <= 512 * 1024 * 1024,
                '转译归档 JSON 大小超出 512 MiB 限制')
    comp = _b64d(archive.get('json_b64', ''))
    inflater = _tr_zlib.decompressobj()
    try:
        text = inflater.decompress(comp, size + 1)
    except _tr_zlib.error as exc:
        raise MeshesFormatError('转译归档 zlib 数据损坏：%s' % exc)
    _tr_require(len(text) == size and inflater.eof and not inflater.unused_data and
                not inflater.unconsumed_tail, '转译归档损坏或解压大小不符')
    _tr_require(hashlib.sha256(text).hexdigest() == archive.get('json_sha256'),
                '转译归档 JSON 哈希不符')
    original = json.loads(text.decode('utf-8'))
    _tr_require(isinstance(original, dict), '归档 JSON 无效')
    raw = encode_meshes_json(_tr_normalize_input(original))
    _tr_require(len(raw) == archive.get('meshes_bytes') and
                hashlib.sha256(raw).hexdigest() == archive.get('meshes_sha256'),
                '转译归档 meshes 校验失败；拒绝恢复')
    return original, raw


def translate_meshes_json(decoded, target_version, mode='strict', keep_archive=True,
                          restore_original=True):
    """Translate this decoder's JSON between 0x33..0x3D.

    strict: refuse unrepresentable/unknown fields instead of silently losing data.
    geometry: preserve render triangles, materials, normal XYZ, lighting and skirt
              alpha; rebuild cross-layout terrain in the actual target layout:
              LOD0 TerrainMesh (0x33..0x37), global GEO0 (0x38), or meshlets (0x39+).
              Never substitute terrain with skirt groups. Experimentally migrate
              cloud volume samples and independent proxy/GEO0 material-80 faces.
              Quantization, shading differences and unsupported 0x37 target proxy
              metadata are explicit; the unchanged source remains in the archive.

    No runtime/game compatibility claim is made: prop resources, level definitions,
    shaders, occlusion, collision and other files are version-dependent.
    """
    target = parse_meshes_version(target_version)
    _tr_require(mode in ('strict', 'geometry'), 'mode 必须为 strict 或 geometry')
    _tr_require(isinstance(decoded, dict), '输入 JSON 必须为对象')
    if restore_original and decoded.get('_translation_archive'):
        archived = _tr_find_archive(decoded['_translation_archive'], target)
        if archived is not None:
            original, raw = _tr_archive_unpack(archived)
            # Do not silently discard edits made after translation.
            receipt = decoded.get('_translation_restore') or decoded.get('_translation') or {}
            current_raw = encode_meshes_json(_tr_normalize_input(decoded))
            expected = receipt.get('output_meshes_sha256', receipt.get('meshes_sha256'))
            _tr_require(hashlib.sha256(current_raw).hexdigest() == expected,
                        '转译后 JSON 已编辑：自动归档恢复会丢失新编辑。请使用 --no-restore 进行重新转译')
            if receipt.get('output_json_content_sha256'):
                _tr_require(_tr_content_hash(decoded) == receipt['output_json_content_sha256'],
                            '转译后 JSON 内容或注释已编辑；请使用 --no-restore，避免自动恢复丢失新编辑')
            restored = copy.deepcopy(original)
            restored['_translation_restore'] = {
                'restored_archive': True, 'target_version': target,
                'meshes_sha256': hashlib.sha256(raw).hexdigest(),
                'note': '恢复归档的原始 JSON/meshes；不是反推有损数据。',
                'output_json_content_sha256': _tr_content_hash(restored)}
            return restored

    source = _tr_normalize_input(decoded)
    version = source['version']
    source_raw = encode_meshes_json(source)
    report = {'source_version': version, 'source_version_hex': '0x%02X' % version,
              'target_version': target, 'target_version_hex': '0x%02X' % target,
              'mode': mode, 'operations': [], 'warnings': [], 'omitted': [],
              'source_meshes_sha256': hashlib.sha256(source_raw).hexdigest(),
              'runtime_tested': False, 'runtime_compatible_guaranteed': False,
              'collision_added': False}
    operations, warnings, omitted = report['operations'], report['warnings'], report['omitted']
    if version == target:
        fresh = _tr_quiet_decode(source_raw)
        _tr_require(encode_meshes_json(fresh) == source_raw, '同版本二进制往返校验失败')
        previous, prior = source.get('_translation_archive'), source.get('_translation') or {}
        unchanged = (prior.get('output_meshes_sha256') == hashlib.sha256(source_raw).hexdigest() and
                     prior.get('output_json_content_sha256') == _tr_content_hash(source))
        if previous and not unchanged and keep_archive:
            source['_translation_archive'] = _tr_archive_pack(decoded, source_raw)
        source.pop('_translation_restore', None)
        source['_translation'] = report
        report.update({'same_version': True, 'binary_roundtrip': True,
                       'output_meshes_sha256': hashlib.sha256(source_raw).hexdigest()})
        report['output_json_content_sha256'] = _tr_content_hash(source)
        return source

    _tr_require(not source.get('lod0_undecoded'), 'LOD0 未解码，无法转译；源文件仍可同版本往返')
    geo = source.get('geo0') or {}
    _tr_require(not geo.get('opaque'), 'GEO0 布局未知，无法保留渲染几何；拒绝转译')
    issues = _tr_source_issues(source)
    if mode == 'strict' and issues:
        raise MeshesFormatError('严格转译被阻止：' + '；'.join(issues))
    if issues:
        omitted.extend(issues)
    cloud = source.get('cloud') or {}
    portable_cloud = _tr_cloud_is_portable(cloud, version, target)
    if not portable_cloud and mode == 'strict':
        raise MeshesFormatError('严格转译被阻止：体积云布局 %s → %s 不能保证无损转换。'
                                'geometry 可显式尝试云采样/真实云网格转换并归档原数据；'
                                '0x37 目标 proxy 元数据仍未实现，不能保证云效果/碰撞等价' %
                                (_tr_cloud_family(version), _tr_cloud_family(target)))
    native_terrain = ((version <= 0x37 and target <= 0x37) or
                      (version == 0x38 and target == 0x38) or
                      (version >= 0x39 and target >= 0x39))
    has_terrain = bool(source.get('terrain_patches')) or bool(
        geo.get('vert_count') or geo.get('chunks') or geo.get('draw_groups'))
    if has_terrain and not native_terrain and mode == 'strict':
        destination = ('重建目标 GEO0 地形和材质绘制表' if target >= 0x38 else
                       '重建目标原生 LOD0 静态地形及空间分块')
        raise MeshesFormatError('严格转译被阻止：LOD0/0x38 GEO0/现代 GEO0 的地形、材质绘制与'
                                '剔除表不能无损互换。geometry 模式会' + destination + '，并归档源表')
    source_metr = _b64d(source.get('metr') or '')
    empty_metr = struct.pack('<2I', 1, 64) + bytes(56)
    if mode == 'strict' and target < 0x36 and source_metr and source_metr != empty_metr:
        raise MeshesFormatError('严格转译被阻止：目标版本没有 METR，非空烘焙统计无对应字段')
    if mode == 'geometry':
        _tr_require(keep_archive, 'geometry 模式必须保留原始归档，不能静默丢弃不兼容字段')

    out = copy.deepcopy(source)
    for key in ('_remaining', 'lod0_undecoded', '_translation', '_translation_archive',
                '_translation_restore'):
        out.pop(key, None)
    out['format'], out['version'], out['version_hex'] = 'sky-meshes-v2', target, '0x%02X' % target
    out['version_hint'] = _version_name(target)
    out['cloud_mesh'] = {'version': target, 'groups': [],
                         'mesh_16': copy.deepcopy((source.get('cloud_mesh') or {}).get('mesh_16'))}
    out['container'] = {'order': ['GEO0', 'LOD0', 'METR'] if target >= 0x38 else
                        (['LOD0', 'METR'] if target >= 0x36 else ['LOD0'])}
    if target >= 0x36:
        out['metr'] = _b64e(source_metr if mode == 'strict' and source_metr else empty_metr)
    else:
        out['metr'] = None
    operations.append('重建目标 LVL0 目录、填充、GEO0/METR 存在条件；清除旧压缩缓存')
    if source_metr:
        if mode == 'strict' and target >= 0x36:
            operations.append('METR 格式相同，原样保留全部统计字段')
        else:
            operations.append('旧 METR 烘焙统计仅归档；目标使用空统计或无 METR')
            if source_metr != empty_metr:
                omitted.append('源 METR 非空烘焙统计（仅归档，不写入目标统计）')
    warnings.append('meshes 格式版本不是游戏资源版本；泛型网格引用需配套目标游戏资源/关卡文件')
    report['generic_meshes_preserved'] = len(out.get('generic_meshes') or [])
    report['occluder_preserved'] = True
    if native_terrain:
        out['terrain_patches'] = copy.deepcopy(source.get('terrain_patches') or []) if target <= 0x37 else []
        out['geo0'] = copy.deepcopy(source.get('geo0')) if target >= 0x39 else None
        if target >= 0x38 and not out['geo0']:
            out['geo0'] = _tr_empty_geo(target)
        if out.get('geo0'):
            trailing = _b64d(out['geo0'].get('trailing_b64', ''))
            if mode == 'geometry' or trailing == struct.pack('<I', 0):
                out['geo0'].pop('trailing_b64', None)
            if target >= 0x3D and not out['geo0'].get('trailing_b64'):
                out['geo0']['trailing_b64'] = _b64e(struct.pack('<I', 0))
        operations.append('保留同布局地形及其原始剔除/子块表；按目标条件写入空导航区域计数')
    else:
        out['terrain_patches'] = []
        out['geo0'] = _tr_empty_geo(target) if target >= 0x38 else None

    # Geometry migration is tracked independently of volume layout. Clouds in
    # GEO0 and the old independent proxy are not skirt groups and must not be
    # hidden by the ordinary terrain digest or created as collision geometry.
    cloud_volume_converted = cloud_geometry_changed = cloud_unsupported = False
    cloud_geometry_inserted_triangles = cloud_geometry_removed_triangles = 0
    cloud_expected = None
    terrain_rebuild_surfaces = None
    terrain_full36_excludes_cloud = False
    source_proxy = None
    if cloud.get('present') and version < 0x38 and target >= 0x38:
        source_proxy = _tr_cloud_proxy_surface(cloud, version)
    _, source_cloud_faces = _tr_cloud_split_surfaces(_tr_geometry_surfaces(source))

    if portable_cloud:
        out['cloud'] = copy.deepcopy(cloud) or {'present': False}
        out['cloud']['version'] = target
        if target <= 0x35:
            out['cloud'].pop('trail', None)
        elif target == 0x36 and out['cloud'].get('present'):
            out['cloud'].setdefault('trail', 0)
        operations.append('体积云为相同布局，原样保留压缩采样数据')
        report['cloud_output'] = 'native_volume_or_empty'
    else:
        converted_cloud, volume_stats = _tr_convert_cloud_volume(cloud, version, target)
        report['cloud_volume_conversion'] = volume_stats
        if converted_cloud is None:
            out['cloud'] = {'present': False, 'version': target}
            cloud_unsupported = True
            report['cloud_output'] = 'unsupported_target_proxy_archive_only'
            omitted.append('目标 0x37 独立云 proxy 元数据生成规则未确认；体积云仅归档')
            warnings.append('0x37 云转换尚未实现：不伪造 proxy 元数据；目标不含源体积云，'
                            '即使材质80几何仍保留，也不保证云可见')
        else:
            out['cloud'] = converted_cloud
            cloud_volume_converted = True
            report['cloud_output'] = 'converted_volume'
            operations.append('转换云粗距离网格及 LZ4 布局，保留原点、维度、brick 顺序、'
                              '距离/光照分辨率和云材质浮点字段')
            distance_stats = volume_stats['distance']
            if distance_stats['converted'] and (distance_stats.get('quantized_samples') or
                                                 distance_stats['saturated_samples']):
                omitted.append('云距离采样量化/饱和的原始精度（%d 量化、%d 饱和；原采样归档）' %
                               (distance_stats.get('quantized_samples', 0),
                                distance_stats['saturated_samples']))
                warnings.append('云距离跨无符号/有符号格式：CPU 最大误差 %.6g、GPU 最大误差 %.6g；'
                                'CPU /128 与 R8_SNORM GPU /127 差异已记录，不保证体积碰撞等价' %
                                (distance_stats['max_abs_cpu_error'], distance_stats['max_abs_gpu_error']))
            if volume_stats['coarse']['converted']:
                warnings.append('云粗距离按 shader 公式转换，最大距离误差 %.6g；'
                                '旧/新 CPU 非活动 brick 回退行为不保证等价' %
                                volume_stats['coarse']['max_abs_shader_distance_error'])
            warnings.append('保留源云光照压缩块；同一 meshes 格式可能配不同游戏 shader，'
                            '旧 Dawn 与新版光照解码不同，不保证亮度或云雾效果等价')
            if volume_stats.get('source_proxy_metadata_archived'):
                omitted.append('源 0x37 proxy 专用元数据和附加顶点通道（目标不使用；原字节归档）')
            if volume_stats.get('source_trail_archived'):
                omitted.append('源云非零 trail 专用字段（目标生成字段不等价；原字节归档）')

    if cloud_volume_converted and version <= 0x36 and target <= 0x36:
        cloud_geometry_changed = bool((cloud.get('v6') or [0, 0, 0])[2])
    if cloud_volume_converted and target >= 0x38 and source_proxy:
        cloud_expected, bake_stats = _tr_cloud_bake_proxy(source_proxy, out['cloud'], target)
        report['cloud_proxy_bake'] = bake_stats
        ordinary, existing_cloud = _tr_cloud_split_surfaces(_tr_geometry_surfaces(source))
        terrain_rebuild_surfaces = ordinary + existing_cloud + [cloud_expected]
        cloud_geometry_inserted_triangles = len(cloud_expected[2]) // 3
        cloud_geometry_changed = True
        report['cloud_output'] = 'converted_volume_and_geo0_proxy'
        operations.append('旧独立云代理 %d 三角形写入真正的 GEO0 材质80云绘制表；'
                          '位置/packed normal XYZ/绕序/重复面保留，不新增裙边或碰撞' %
                          cloud_geometry_inserted_triangles)
        warnings.append('新增云网格 light0/light1 取目标编码 atlas 三线性采样；'
                        'light2 取源法线偏置字节，不声明重烘焙光照或跨版外观等价')
        if bake_stats['inactive_bin_vertices']:
            warnings.append('%d 个源云顶点处于非活动 brick，使用显式目标 shader 默认光照' %
                            bake_stats['inactive_bin_vertices'])
    elif cloud_volume_converted and target <= 0x36 and version >= 0x37:
        source_surfaces = _tr_geometry_surfaces(source)
        ordinary, modern_cloud = _tr_cloud_split_surfaces(source_surfaces) if version >= 0x38 else (source_surfaces, [])
        if version == 0x37:
            proxy = _tr_cloud_proxy_surface(cloud, version)
            if proxy:
                modern_cloud.append(proxy)
        report['cloud_legacy_proxy'] = _tr_cloud_make_legacy_proxy(modern_cloud, out['cloud'])
        if version >= 0x38 and source_cloud_faces:
            # The old volume's 16B proxy, not TerrainMesh material80, is the
            # destination of modern cloud render faces. Ordinary vertices retain
            # full 36B equality; downgrade-only baked lanes remain archived.
            terrain_rebuild_surfaces = ordinary
            terrain_full36_excludes_cloud = True
            cloud_geometry_removed_triangles = sum(len(s[2]) // 3 for s in source_cloud_faces)
            omitted.append('现代云材质/权重/12B baked 顶点属性（旧 16B 代理无对应字段；原字节归档）')
        if version == 0x37 and (cloud.get('proxy') or {}).get('vert_count'):
            omitted.append('0x37 云代理附加4B顶点通道及 normal.w（旧 16B 代理无等价字段；源数据归档）')
        cloud_geometry_changed = bool(modern_cloud)
        report['cloud_output'] = 'converted_volume_and_legacy_proxy'
        operations.append('现代云绘制面/0x37 代理重建为旧 16B 独立云代理；'
                          '保持真实位置/法线XYZ/有向面，不写入普通地形碰撞路径')
    report['cloud_geometry_inserted_in_geo0_triangles'] = cloud_geometry_inserted_triangles
    report['cloud_geometry_removed_from_terrain_triangles'] = cloud_geometry_removed_triangles
    report['cloud_conversion_experimental'] = cloud_volume_converted

    destination_stride = 40 if target <= 0x38 else 36
    groups = out['cloud_mesh']['groups']
    alpha_quantized_vertices = 0
    for i, group in enumerate((source.get('cloud_mesh') or {}).get('groups') or []):
        raw, stride, indices, label = _tr_skirt_surface(group, version, 'skirt_%d' % i)
        if stride == 40 and destination_stride == 36:
            normal_extra = any(raw[i + 15] not in (0, raw[i + 39]) for i in range(0, len(raw), 40))
            if normal_extra:
                if mode == 'strict':
                    raise MeshesFormatError('严格转译被阻止：旧裙边 normal.w 非零且与 alpha 不同，目标无法同时表示')
                omitted.append('旧裙边独立 normal.w 分量（目标该分量用于 alpha；源值在归档中）')
            user_xyz = any(raw[at:at + 3] != b'\x00\x00\x00' for at in range(36, len(raw), 40))
            if user_xyz:
                if mode == 'strict':
                    raise MeshesFormatError('严格转译被阻止：40B 裙边 userData1.xyz 非零，36B 目标无对应字段')
                if '旧裙边 userData1.xyz（目标 36B 顶点无对应字段）' not in omitted:
                    omitted.append('旧裙边 userData1.xyz（目标 36B 顶点无对应字段）')
        converted = _tr_convert_skirt_vertices(raw, stride, destination_stride, True)
        converted, quantized = _tr_convert_skirt_alpha(converted, destination_stride, version, target)
        if quantized and mode == 'strict':
            raise MeshesFormatError('严格转译被阻止：现代裙边 alpha 无法精确表示为旧版步进值；'
                                    'geometry 可显式量化并归档原字节')
        alpha_quantized_vertices += quantized
        # Existing uint16 skirts need no vertex compaction. Keep even unreferenced
        # vertices and exact index/vertex order in strict as well as geometry mode.
        groups.append(_tr_make_skirt(converted, indices, target))
    operations.append('裙边顶点 40B/36B 与法线 alpha 分量转换；raw uint16/meshopt 索引重新编码')
    if (version <= 0x3A) != (target <= 0x3A):
        operations.append('按 0.35.1 LoadLod 兼容公式转换裙边 alpha：旧 int8×10、钳位、减128；'
                          '现代目标存储偏移后字节')
    if alpha_quantized_vertices:
        omitted.append('现代裙边 alpha 降至旧版量化步进：%d 个顶点无法字节等价（原字节归档）' %
                       alpha_quantized_vertices)
        warnings.append('裙边透明度降级发生显式量化；每个值的目标不透明度误差不超过 5/255')
    report['skirt_alpha_quantized_vertices'] = alpha_quantized_vertices
    report['skirt_alpha_max_opacity_error'] = 5 / 255 if alpha_quantized_vertices else 0.0

    terrain_triangles = 0  # Kept for old report readers; terrain is NEVER moved to skirts.
    geo0_triangles, lod0_triangles = 0, 0
    rebuilt_geo0 = rebuilt_lod0 = False
    if (has_terrain and not native_terrain) or terrain_rebuild_surfaces is not None:
        surfaces = (terrain_rebuild_surfaces if terrain_rebuild_surfaces is not None else
                    _tr_geometry_surfaces(source))
        triangle_count = sum(len(indices) // 3 for raw, stride, indices, label in surfaces)
        if target >= 0x38:
            out['geo0'] = (_tr_make_geo038(surfaces) if target == 0x38 else
                           _tr_make_modern_geo0(surfaces, target))
            rebuilt_geo0 = True
            geo0_triangles = triangle_count
            operations.append('跨布局地形 %d 三角形重建为真正的目标 GEO0，保留原始 36B 顶点、'
                              '有向面、材质 ID 和权重；不混入 LOD0 裙边' % geo0_triangles)
            operations.append('0x38 重建独立深度/材质/云绘制表；0x39+ 重建权重筛选材质子块；'
                              '重算包围盒并关闭重建分块的法线锥剔除')
            omitted.append('跨布局原地形剔除/材质 pass 分块表（目标重新构建 GEO0 表；源表在归档中）')
            if target >= 0x3D:
                warnings.append('源跨布局地图没有目标 wayfinder 导航区域；目标 GEO0 写入区域数 0')
        else:
            out['terrain_patches'] = _tr_make_legacy_terrain(surfaces, target)
            rebuilt_lod0 = True
            lod0_triangles = triangle_count
            operations.append('跨布局地形 %d 三角形重建为目标原生 LOD0 TerrainMesh；保留原始 36B 顶点、'
                              '有向面、材质 ID 和权重，按 uint16 限制拆分，不新增裙边' % lod0_triangles)
            operations.append('每个旧地形网格建立单个完整空间单元，走静态材质绘制路径；'
                              '无源 TerrainBlobGroup 引用，不伪造动态细分或碰撞实例')
            omitted.append('跨布局原地形空间剔除/动态细分/材质分块表（目标按静态几何重建；源表在归档中）')
            warnings.append('重建旧地形不添加碰撞实例，也不恢复源动态细分；'
                            '目标 TerrainBlobGroup、碰撞和 LOD 须配套资源并在游戏内复测')
        warnings.append('原生地形加载/材质表通过离线结构校验不等于游戏内验证；'
                        '地形 shader、纹理、碰撞和 LOD 行为不保证跨游戏版本等价')
    out['bounds'] = _tr_bounds(out)
    if out.get('_cube'):
        out['_cube']['translation_note'] = '原 group_index/vertex_stride 是源版本标记；目标实际组见 _translation'
    out['_meta'] = dict(out.get('_meta') or {}, decoder='universal-v3',
                        translated_from_version=version, modified=True)
    if keep_archive:
        # Reusing a valid earlier archive avoids recursively nesting megabytes.
        previous = decoded.get('_translation_archive')
        previous_report = decoded.get('_translation') or {}
        unchanged = (previous_report.get('output_meshes_sha256') ==
                     hashlib.sha256(source_raw).hexdigest() and
                     previous_report.get('output_json_content_sha256') == _tr_content_hash(decoded))
        if previous and unchanged:
            _tr_archive_unpack(previous)
            out['_translation_archive'] = copy.deepcopy(previous)
        else:
            # If the intermediate JSON was edited, archive that edited source as
            # well. The older archive is inside it; bounded history lookup can
            # still recover the original version. Never claim unarchived edits
            # are reversible merely because an earlier source snapshot exists.
            out['_translation_archive'] = _tr_archive_pack(decoded, source_raw)
    out['_translation'] = report
    report.update({'terrain_triangles_moved_to_skirts': terrain_triangles,
                   'terrain_triangles_rebuilt_in_geo0': geo0_triangles,
                   'terrain_triangles_rebuilt_in_lod0': lod0_triangles,
                   'terrain_output': ('rebuilt_geo0_0x38' if rebuilt_geo0 and target == 0x38 else
                                      'rebuilt_geo0' if rebuilt_geo0 else
                                      'rebuilt_lod0' if rebuilt_lod0 else 'native_or_empty'),
                   'terrain_skirt_fallback_used': False,
                   'target_skirt_groups': len(groups), 'archive_embedded': keep_archive,
                   'data_loss_in_target_meshes': bool(omitted),
                   'original_data_recoverable_from_archive': keep_archive,
                   'archive_source_version': (out.get('_translation_archive') or {}).get('source_version')})

    # Validation exercises the real encoder AND a fresh decode in the target
    # layout. A wrong version tag or reused old LOD0 cannot satisfy these checks.
    report['lod0_loader_limit_verified'] = _tr_check_lod0_limit(encode_lod0(out, target))
    target_raw = encode_meshes_json(out, force_recompress=True)
    verified = _tr_quiet_decode(target_raw)
    _tr_require(verified['version'] == target and not verified.get('lod0_undecoded') and
                not verified.get('_remaining'), '目标 meshes 未完整按目标版本解码')
    _tr_require(verified['container']['order'] == out['container']['order'], '目标分段顺序错误')
    if rebuilt_geo0:
        report['geo0_render_tables_verified'] = (_tr_validate_geo038_render_tables(verified['geo0'])
                                                if target == 0x38 else
                                                _tr_validate_geo0_render_tables(verified['geo0']))
    if rebuilt_lod0:
        report['lod0_terrain_tables_verified'] = _tr_validate_legacy_terrain(verified['terrain_patches'], target)
    _tr_require(len((verified.get('cloud_mesh') or {}).get('groups') or []) ==
                len((source.get('cloud_mesh') or {}).get('groups') or []),
                '转译不得新增地形裙边组')
    _tr_require(_tr_generic_bytes(verified) == _tr_generic_bytes(source), '泛型资源列表转译校验失败')
    _tr_require(_tr_occluder_bytes(verified) == _tr_occluder_bytes(source), '遮挡网格转译校验失败')
    # Include every explicitly prepared 36B proxy vertex in the full-attribute
    # expectation. Only a modern-to-16B cloud downgrade narrows cloud fields;
    # ordinary terrain/skirt bytes are never exempted from comparison.
    before_digest = _tr_surface_digest(source, alpha_target_version=target,
                                       exclude_cloud_material=terrain_full36_excludes_cloud,
                                       extra_surfaces=[cloud_expected] if cloud_expected else None)
    after_digest = _tr_surface_digest(verified, exclude_cloud_material=terrain_full36_excludes_cloud)
    _tr_require(before_digest == after_digest,
                '转译几何/绕序/材质/光照/alpha 校验不一致')
    if cloud_geometry_changed:
        report['cloud_geometry_verified'] = _tr_cloud_render_digest(verified)
        _tr_require(report['cloud_geometry_verified'] == _tr_cloud_render_digest(source),
                    '转译云代理/GEO0 位置、法线XYZ、绕序或重复面校验不一致')
        report['ordinary_geometry_full36_preserved'] = True
        report['cloud_full36_attributes_excluded_on_legacy_downgrade'] = terrain_full36_excludes_cloud
    if alpha_quantized_vertices:
        report['source_geometry_before_alpha_quantization'] = _tr_surface_digest(source)
        report['geometry_verified_with_declared_alpha_quantization'] = True
    _tr_require(encode_meshes_json(verified) == target_raw, '目标 meshes 二次往返不是逐字节一致')
    if portable_cloud or cloud_volume_converted:
        cloud_writer = BinaryStreamWriter()
        encode_cloud_section(cloud_writer, out['cloud'], target)
        verify_writer = BinaryStreamWriter()
        encode_cloud_section(verify_writer, verified['cloud'], target)
        _tr_require(cloud_writer.to_bytes() == verify_writer.to_bytes(), '体积云往返校验失败')
        report['cloud_section_binary_roundtrip'] = True
    if cloud_volume_converted:
        actual_samples, actual_b, actual_a = _tr_cloud_unpack(verified['cloud'], target)
        expected_samples, expected_b, expected_a = _tr_cloud_unpack(out['cloud'], target)
        _tr_require((actual_b, actual_a) == (expected_b, expected_a) and
                    actual_samples == expected_samples, '目标云采样解压校验不一致')
        report['cloud_volume_samples_verified'] = {key: {'bytes': len(value),
                    'sha256': hashlib.sha256(value).hexdigest()} for key, value in actual_samples.items()}
    report.update({'binary_roundtrip': True, 'fresh_target_decode': True,
                   'geometry_verified': before_digest,
                   'output_meshes_sha256': hashlib.sha256(target_raw).hexdigest(),
                   'output_meshes_bytes': len(target_raw)})
    # Store canonical target JSON: no source-layout keys or stale compression.
    # Preserve only annotations and the reversible archive from the working dict.
    for key, value in out.items():
        if key.startswith('_'):
            verified[key] = value
    report['output_json_content_sha256'] = _tr_content_hash(verified)
    return verified


def _tr_clean_path(value):
    path = value.strip()
    if len(path) > 1 and path[0] == path[-1] and path[0] in ('"', "'"):
        path = path[1:-1]
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _tr_same_path(first, second):
    # Include junction/symlink/8.3/hard-link aliases, not just the typed spelling.
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _tr_atomic_write(path, data, overwrite=False):
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    _tr_require(os.path.isdir(folder), '输出目录不存在：' + folder)
    if not overwrite and os.path.lexists(path):
        raise MeshesFormatError('输出已存在，未覆盖：%s；CLI 可显式指定 --overwrite' % path)
    fd, tmp = _tr_tempfile.mkstemp(prefix='.meshes_translate_', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if overwrite:
                os.replace(tmp, path)
            elif os.name == 'nt':
                # Windows rename refuses an existing target atomically. An exists
                # check followed by replace could clobber a concurrent writer.
                os.rename(tmp, path)
            else:
                os.link(tmp, path)
                os.unlink(tmp)
        except FileExistsError:
            raise MeshesFormatError('输出在写入期间已被创建，未覆盖：' + path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def translate_json_file(input_path, target_version, output_path=None, mode='strict',
                        meshes_path=None, overwrite=False, restore_original=True):
    input_path = _tr_clean_path(input_path)
    target = parse_meshes_version(target_version)
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.to_0x%02X.json' % target
    output_path = _tr_clean_path(output_path)
    _tr_require(not _tr_same_path(output_path, input_path), '转译输出不得覆盖输入 JSON')
    if meshes_path:
        meshes_path = _tr_clean_path(meshes_path)
        _tr_require(not _tr_same_path(meshes_path, input_path) and
                    not _tr_same_path(meshes_path, output_path), '输出文件路径必须互不相同')
    outputs = [output_path] + ([meshes_path] if meshes_path else [])
    for path in outputs:
        _tr_require(os.path.isdir(os.path.dirname(path)), '输出目录不存在：' + path)
        _tr_require(overwrite or not os.path.lexists(path), '输出已存在，未覆盖：' + path)
    with open(input_path, 'r', encoding='utf-8-sig') as handle:
        original = json.load(handle)
    translated = translate_meshes_json(original, target, mode=mode,
                                       restore_original=restore_original)
    # Serialize/reload JSON before final encoding: verify the actual JSON shape,
    # not an in-memory-only conversion result. Strict JSON disallows NaN/Infinity.
    text = json.dumps(translated, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    reread = json.loads(text.decode('utf-8'))
    raw = encode_meshes_json(reread)
    report = reread.get('_translation_restore') or reread.get('_translation') or {}
    expected = report.get('output_meshes_sha256', report.get('meshes_sha256'))
    _tr_require(expected == hashlib.sha256(raw).hexdigest(), '序列化 JSON 后编码哈希改变')
    _tr_atomic_write(output_path, text, overwrite)
    if meshes_path:
        _tr_atomic_write(meshes_path, raw, overwrite)
    print('转译 JSON：%s  (目标 0x%02X)' % (output_path, target))
    if meshes_path:
        print('目标 meshes：%s  (%d bytes)' % (meshes_path, len(raw)))
    for warning in report.get('warnings') or []:
        print('  [注意] ' + warning)
    for omission in report.get('omitted') or []:
        print('  [仅归档，不写入目标 meshes] ' + omission)
    print('已通过目标解码和二进制往返校验；不代表已通过游戏内运行测试。')
    return translated

def _interactive_version():
    print('\n可选 meshes 解码/编码格式（不是游戏版本号）：')
    for i, version in enumerate(range(0x33, 0x3E), 1):
        layout = ('LOD0 旧地形 / raw 裙边' if version <= 0x36 else
                  'LOD0 旧地形 / meshopt 裙边' if version == 0x37 else
                  'GEO0 全局索引地形' if version == 0x38 else
                  'GEO0 分块地形 / 36B meshopt 裙边')
        print('  %2d. 0x%02X  %s' % (i, version, layout))
    value = input('选择序号 1～11，或输入 0x33～0x3D（回车取消）：').strip()
    if not value:
        return None
    if value.isdecimal() and 1 <= int(value) <= 11:
        return 0x32 + int(value)
    return parse_meshes_version(value)


def _interactive_output(prompt, default):
    value = input('%s [%s]：' % (prompt, default)).strip()
    path = _tr_clean_path(value or default)
    if os.path.exists(path):
        answer = input('文件已存在，是否覆盖？输入 YES 确认：').strip()
        if answer != 'YES':
            raise MeshesFormatError('已取消，未覆盖文件')
    return path


def interactive_main():
    """No-argument menu. Function 3 is JSON version translation, not OBJ import."""
    while True:
        print('''
Sky meshes 解码 / 编码 / JSON 版本转译  (0x33～0x3D)
  1. meshes → JSON
  2. JSON → meshes
  3. JSON 版本转译（选择目标版本，可同时输出 meshes）
  4. JSON → OBJ
  5. OBJ 导入 JSON（原有功能，仅旧地形布局）
  6. 合并两个 JSON（原有版本限制不变）
  7. 查看 meshes 信息
  0. 退出
''')
        try:
            choice = input('选择功能：').strip()
            if choice in ('0', 'q', 'quit', 'exit'):
                return
            if choice == '1':
                value = input('输入 meshes 文件路径（支持拖入/带引号，回车取消）：').strip()
                if not value:
                    continue
                inp = _tr_clean_path(value)
                out = _interactive_output('输出 JSON 路径', os.path.splitext(inp)[0] + '.json')
                _tr_require(not _tr_same_path(inp, out), '不能覆盖输入 meshes')
                meshes_to_json(inp, out)
            elif choice == '2':
                value = input('输入 JSON 文件路径（回车取消）：').strip()
                if not value:
                    continue
                inp = _tr_clean_path(value)
                out = _interactive_output('输出 meshes 路径', os.path.splitext(inp)[0] + '.meshes')
                _tr_require(not _tr_same_path(inp, out), '不能覆盖输入 JSON')
                json_to_meshes(inp, out)
            elif choice == '3':
                value = input('输入已解码的 JSON 文件路径（回车取消）：').strip()
                if not value:
                    continue
                inp = _tr_clean_path(value)
                with open(inp, 'r', encoding='utf-8-sig') as handle:
                    source = json.load(handle)
                _tr_require(isinstance(source, dict), '输入 JSON 根节点必须为对象')
                current = source.get('version', source.get('version_hex', 0x36))
                print('源 meshes 格式：0x%02X' % parse_meshes_version(current))
                target = _interactive_version()
                if target is None:
                    continue
                restore_original = True
                if source.get('_translation_archive'):
                    print('输入 JSON 带有原始转译归档：')
                    print('  1. 自动恢复目标版本的归档（默认；若当前 JSON 已编辑，会阻止恢复）')
                    print('  2. 转译当前 JSON 数据（保留新编辑，不自动合并旧归档数据）')
                    archive_choice = input('归档处理 [1]：').strip() or '1'
                    _tr_require(archive_choice in ('1', '2'), '归档处理应为 1 或 2')
                    restore_original = archive_choice == '1'
                print('  1. 严格转译（默认）：不支持无损表示的字段将报错，不丢弃')
                print('  2. 渲染几何兼容：重建原生 LOD0/GEO0；试转真实云网格/采样，量化及0x37限制需归档')
                mode_choice = input('选择模式 [1]：').strip() or '1'
                _tr_require(mode_choice in ('1', '2'), '模式应为 1 或 2')
                mode = 'geometry' if mode_choice == '2' else 'strict'
                if mode == 'geometry':
                    print('兼容转译不保证体积云、地形 shader、碰撞、LOD 与源版本一致。')
                    print('原始 JSON 嵌入归档，可还原源版本。输入文件不会改写。')
                    if input('确认采用兼容转译？输入 YES：').strip() != 'YES':
                        continue
                out = _interactive_output('转译 JSON 路径',
                                          os.path.splitext(inp)[0] + '.to_0x%02X.json' % target)
                output_mesh = None
                if input('同时输出目标 meshes？[y/N]：').strip().lower() in ('y', 'yes'):
                    output_mesh = _interactive_output('目标 meshes 路径', os.path.splitext(out)[0] + '.meshes')
                translate_json_file(inp, target, out, mode=mode, meshes_path=output_mesh,
                                    overwrite=True, restore_original=restore_original)
            elif choice == '4':
                inp = _tr_clean_path(input('输入 JSON 路径：'))
                folder = _tr_clean_path(input('输出目录 [obj_export]：') or 'obj_export')
                json_to_obj(inp, folder)
            elif choice == '5':
                obj = _tr_clean_path(input('输入 OBJ 路径：'))
                inp = _tr_clean_path(input('输入作为模板的 JSON 路径：'))
                out = _interactive_output('输出 JSON 路径', os.path.splitext(inp)[0] + '.imported.json')
                import_obj_to_json(obj, inp, out)
            elif choice == '6':
                a = _tr_clean_path(input('第一个 JSON 路径：'))
                b = _tr_clean_path(input('第二个 JSON 路径：'))
                out = _interactive_output('合并输出路径', os.path.splitext(a)[0] + '.merged.json')
                merge_json(a, b, out)
            elif choice == '7':
                meshes_info(_tr_clean_path(input('输入 meshes 路径：')))
            else:
                print('请输入 0～7。')
        except (EOFError, KeyboardInterrupt):
            print('\n已退出，未完成的操作已取消。')
            return
        except (OSError, ValueError, TypeError, KeyError, MeshesFormatError,
                EOFError, struct.error) as exc:
            print('\n[失败] %s' % exc)


_print_usage_v2 = print_usage
_main_v2 = main


def print_usage():
    _print_usage_v2()
    print('''新增功能：
  无参数启动：显示中文菜单，功能 3 为 JSON 版本转译。
  python "Meshes←→json与编辑(半成品).py" translate input.json 0x3D [output.json]
      [--mode strict|geometry] [--meshes output.meshes] [--overwrite] [--no-restore]

  strict（默认）：不能无损表达的数据报错，不静默丢弃。
  geometry：跨 0x33～0x3D 保留可解码渲染三角形/材质/法线/光照/alpha，
            0x33～0x37 重建原生 LOD0 静态地形；0x38 重建全局 GEO0 材质表；
            0x39+ 重建 GEO0 分块/子块，不把地形转成裙边。新 LOD0 限制 12 MiB。
            裙边 alpha 按加载公式转换；现代值降级可能量化（最大误差 5/255）。
            实验性转换云采样/真实独立代理到 GEO0 材质80，或现代云降至旧16B代理。
            距离量化/饱和与光照差异会报告；0x37 目标 proxy 元数据仍不支持，只归档。
            专用元数据和原始精度保存在 JSON 归档；不保证游戏内云可见、外观或碰撞等价。
            转回归档的源版本且未修改目标数据时，可恢复原始 JSON/meshes。
            若目标 JSON 已修改，使用 --no-restore 转译新数据，不能自动无损合并归档。
  转译 JSON 可继续使用 encode 命令或菜单 2 编码回目标 meshes。
  这不是只改 version；不承诺跨游戏资源/关卡兼容，不保证游戏内运行和碰撞。
''')


def main():
    if len(sys.argv) < 2:
        return interactive_main()
    if sys.argv[1].lower() in ('menu', 'interactive'):
        return interactive_main()
    if sys.argv[1].lower() in ('translate', 'transcode', 'translate-json', '3', '转译'):
        import argparse
        parser = argparse.ArgumentParser(description='Sky meshes JSON 版本转译 (0x33～0x3D)')
        parser.add_argument('input', help='本脚本解码的 JSON 路径')
        parser.add_argument('version', help='目标版本，如 0x3D / 3D / 61')
        parser.add_argument('output', nargs='?', help='输出 JSON；默认 source.to_0xXX.json')
        parser.add_argument('--mode', choices=('strict', 'geometry'), default='strict')
        parser.add_argument('--meshes', help='同时输出经过验证的目标 meshes')
        parser.add_argument('--overwrite', action='store_true', help='允许覆盖已有输出（不允许覆盖输入）')
        parser.add_argument('--no-restore', action='store_true', help='不要恢复归档源数据，转译当前编辑数据')
        args = parser.parse_args(sys.argv[2:])
        try:
            translate_json_file(args.input, args.version, args.output, mode=args.mode,
                                meshes_path=args.meshes, overwrite=args.overwrite,
                                restore_original=not args.no_restore)
        except (MeshesFormatError, OSError, ValueError, TypeError, KeyError,
                EOFError, struct.error) as exc:
            print('[转译失败] %s' % exc, file=sys.stderr)
            raise SystemExit(2)
        return
    return _main_v2()


def _tr_generic_bytes(decoded):
    writer = BinaryStreamWriter()
    encode_generic_meshes(writer, decoded.get('generic_meshes') or [])
    return writer.to_bytes()


def _tr_occluder_bytes(decoded):
    writer = BinaryStreamWriter()
    _encode_cloud_mesh_v2(writer, {'groups': [], 'mesh_16':
                                  (decoded.get('cloud_mesh') or {}).get('mesh_16')}, 0x36)
    return writer.to_bytes()[4:]


def _tr_content_hash(decoded):
    active = {k: v for k, v in decoded.items() if not k.startswith('_translation')}
    text = json.dumps(active, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(text).hexdigest()



def _tr_find_archive(archive, target):
    for _ in range(16):
        _tr_require(isinstance(archive, dict), '转译归档必须为对象')
        if parse_meshes_version(archive.get('source_version')) == target:
            return archive
        original, _raw = _tr_archive_unpack(archive)
        archive = original.get('_translation_archive')
        if not archive:
            return None
    raise MeshesFormatError('转译归档历史超过 16 层；请先单独恢复较早归档')


def _has_geo0(decoded):
    # Include decoded 0x38 draw_groups and valid empty GEO0 dictionaries. Legacy
    # merge/OBJ-import must never silently treat modern data as old LOD terrain.
    return bool(decoded.get('geo0')) or parse_meshes_version(decoded.get('version', 0x36)) >= 0x38



def encode_meshes_json(decoded, force_recompress=False):
    # Old JSON emitted by this SAME script did not include a format version and
    # used float_v5/v5_single for cloud fields. Normalize those names before the
    # existing binary writer, including when restoring an archived legacy JSON.
    _tr_require(isinstance(decoded, dict), '输入 JSON 必须为对象')
    cloud = decoded.get('cloud') or {}
    legacy = ('version' not in decoded or
              ('float_v5' in cloud and 'voxel_scale' not in cloud) or
              ('v5_single' in cloud and 'bpp_dist_amb01' not in cloud))
    out = _tr_normalize_input(decoded) if legacy else decoded
    version = parse_meshes_version(out.get('version', 0x36))
    if out.get('version_hex'):
        _tr_require(parse_meshes_version(out['version_hex']) == version,
                    'version/version_hex 不一致，请使用转译而不是手改版本号')
    if out.get('version') != version:
        out = dict(out, version=version)
    # The underlying encoder checks the raw LOD0 size immediately before any
    # recompression; do not serialize all legacy vertices a second time here.
    return _encode_meshes_json_v2(out, force_recompress=force_recompress)


if __name__ == "__main__":
    main()
