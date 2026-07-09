# Sky: Children of the Light (.mesh)
# Merged & Extended by AI based on Python Tool & Durik256 Plugin
#
# Key improvements:
#  1. ZipPos position/UV now use correct 10-bit / 16-bit dequantization
#     (the old logic read raw bytes directly, which was wrong).
#  2. Creature/skeleton detection no longer relies on the filename (Anim/anc).
#     It uses data flag byte 0x48 instead:
#       - Compressed versions (1C/1E/1F/20): 0x48 != 0 means it has a skeleton.
#       - Uncompressed versions (17/19/1A): 0x48 is the bone count directly.
#     So Elder models that carry a rig are not missed, and static statues/props
#     are not misdetected.
#  3. All versions parse and show the skeleton (Noesis Skeleton) when present,
#     and bind skin weights for animation preview.
#  4. Low versions (17/19/1A) are not LZ4-decompressed; mesh and bones are read
#     in place.
#
# NOTE: This file is intentionally pure ASCII so Noesis's embedded Python
# compiles it regardless of the system locale/encoding.
from inc_noesis import *
import struct
import os
import binascii

def registerNoesisTypes():
    handle = noesis.register("Sky: Children of the Light", ".mesh")
    noesis.setHandlerTypeCheck(handle, noepyCheckType)
    noesis.setHandlerLoadModel(handle, noepyLoadModel)
    # noesis.logPopup()
    return 1

def noepyCheckType(data):
    if len(data) < 4:
        return 0
    return 1

def noepyLoadModel(data, mdlList):
    # ZipPos: route by FILENAME to the dedicated compressed-position handler.
    # (Only the ZipPos path is decided by name; everything else is content-based.)
    if 'zippos' in rapi.getInputName().lower():
        return noepyLoadZipModel(data, mdlList)

    ctx = rapi.rpgCreateContext()
    magic = data[:4]
    filename = rapi.getLocalFileName(rapi.getInputName())
    bones = []

    try:
        if magic in (b'\x17\x00\x00\x00', b'\x18\x00\x00\x00'):
            bones = parse_17(data, filename)
        elif magic in (b'\x19\x00\x00\x00', b'\x1a\x00\x00\x00', b'\x1b\x00\x00\x00'):
            bones = parse_1A(data, filename)
        elif magic in (b'\x1c\x00\x00\x00', b'\x1d\x00\x00\x00'):
            bones = parse_1C(data, filename)
        elif magic == b'\x1e\x00\x00\x00':
            bones = parse_1E(data, filename)
        elif magic == b'\x1f\x00\x00\x00':
            bones = parse_1F20(data, filename, 0x1F)
        elif magic == b'\x20\x00\x00\x00':
            bones = parse_1F20(data, filename, 0x20)
        else:
            hex_magic = binascii.hexlify(magic).decode('ascii')
            print("Unknown magic header: " + hex_magic)
            return 0
    except Exception as e:
        print("Error parsing mesh: " + str(e))
        return 0

    try:
        mdl = rapi.rpgConstructModel()
    except:
        mdl = NoeModel()

    mdl.setModelMaterials(NoeModelMaterials([], [NoeMaterial('default','')]))
    if bones:
        mdl.setBones(bones)

    mdlList.append(mdl)
    return 1

# ======================= Skeleton parsing (data-driven, filename-independent) =======================

def _u32(d, o):
    return struct.unpack_from('<I', d, o)[0]

def has_skeleton_flag(data):
    """Byte 0x48: compressed versions -> non-zero means has skeleton;
    uncompressed 17 -> the value is the bone count directly."""
    if len(data) <= 0x48:
        return 0
    return data[0x48]

def build_bones_from_block(block, bone_count, start):
    """Generic bone-table parse: 132 bytes each = name(64)+matrix(64)+parent(4).
    Works for compressed tail block (1C/1E/1F/20) and inline table (17/19/1A)."""
    bones = []
    p = start
    for x in range(bone_count):
        if p + 132 > len(block):
            break
        name_raw = block[p:p+64].split(b'\x00')[0]
        name = name_raw.decode('ascii', errors='ignore') if name_raw else "bone_{}".format(x)
        p += 64
        mat = NoeMat44.fromBytes(block[p:p+64]).toMat43().inverse()
        p += 64
        parent_idx = struct.unpack_from('<I', block, p)[0] - 1
        p += 4
        bones.append(NoeBone(x, name, mat, None, parent_idx))
    return bones

def parse_tail_bones(tail):
    """Compressed versions (1C/1E/1F/20): the tail data after the compressed block.
    Layout: variable name prefix + header (bone_count at tail[68]) + bone table (from tail[85])."""
    if len(tail) < 85:
        return []
    bone_count = _u32(tail, 68)
    if bone_count <= 0 or bone_count > 4096:
        return []
    return build_bones_from_block(tail, bone_count, 85)

def scan_inline_bones(data, bone_count):
    """Uncompressed versions (17/19/1A): search for 'RigRef' bone names to locate
    the bone table start. 132 bytes each; bone count comes from byte 0x48."""
    if bone_count <= 0 or bone_count > 4096:
        return []
    marker = data.find(b'RigRef')
    if marker < 0:
        return []
    # Name start: walk back from RigRef to the start of the printable string
    s = marker
    while s > 0 and 32 <= data[s-1] < 127:
        s -= 1
    return build_bones_from_block(data, bone_count, s)

# ======================= Per-version parsing =======================

def parse_17(data, filename):
    # Bone count comes directly from byte 0x48 (data-driven, not filename)
    bone_count = has_skeleton_flag(data)
    has_skin = bone_count != 0

    if "StripAnim" in filename:  # case-sensitive tag
        vip = 0x4061; iip = 0x4065; vs = 0x408D
        vnum = struct.unpack('<I', data[vip:vip+4])[0]
        inum = struct.unpack('<I', data[iip:iip+4])[0]
        vbuf_len = vnum * 16

        vbuf = data[vs : vs+vbuf_len]
        gap = vbuf_len // 4
        us = vs + vbuf_len + gap
        uvbuf = data[us : us+vbuf_len]

        idx_s = us + vbuf_len + vnum * 8
        ibuf = data[idx_s : idx_s + inum*4]
    else:
        p01 = data.find(b'\x01')
        if p01 == -1: return []
        vip = p01 + 45; iip = 0x75; vs = 0x9D

        vnum = struct.unpack('<I', data[vip:vip+4])[0]
        inum = struct.unpack('<I', data[iip:iip+4])[0]
        vbuf_len = vnum * 16

        # Sanity check: skinned "creature" files (CharXxxAnim) use another multi-section
        # layout, so fixed offsets give bogus vnum/inum. Then return bones only.
        idx_end = vs + vbuf_len + vbuf_len // 4 + vbuf_len + inum * 4
        plausible = (0 < vnum < 500000 and 0 < inum < 3000000 and idx_end <= len(data))
        if not plausible:
            if has_skin:
                return scan_inline_bones(data, bone_count)
            return []

        vbuf = data[vs : vs+vbuf_len]
        gap = vbuf_len // 4
        us = vs + vbuf_len + gap
        uvbuf = data[us : us+vbuf_len]

        idx_s = us + vbuf_len
        ibuf = data[idx_s : idx_s + inum*4]

    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgBindUV1Buffer(uvbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgCommitTriangles(ibuf, noesis.RPGEODATA_UINT, inum, noesis.RPGEO_TRIANGLE)
    # Bone count from byte 0x48 (data-driven)
    if has_skin:
        return scan_inline_bones(data, bone_count)
    return []

def parse_1A(data, filename):
    bone_count = has_skeleton_flag(data)
    has_skin = bone_count != 0

    vco = 0x66; ico = 0x6A; vs = 0x92
    vnum = struct.unpack('<I', data[vco:vco+4])[0]
    inum = struct.unpack('<I', data[ico:ico+4])[0]
    vbuf_len = vnum * 16

    vbuf = data[vs : vs+vbuf_len]
    gap = vbuf_len // 4
    us = vs + vbuf_len + gap
    uvbuf = data[us : us+vbuf_len]

    # Whether a weight buffer exists is decided by the data flag (not by anim/anc in name)
    idx_s = us + vbuf_len + (vnum * 8 if has_skin else 0)
    ibuf = data[idx_s : idx_s + inum*4]

    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgBindUV1Buffer(uvbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgCommitTriangles(ibuf, noesis.RPGEODATA_UINT, inum, noesis.RPGEO_TRIANGLE)
    if has_skin:
        return scan_inline_bones(data, bone_count)
    return []

def parse_1C(data, filename):
    cs = struct.unpack('<I', data[0x4E:0x52])[0]
    us = struct.unpack('<I', data[0x52:0x56])[0]
    dr = rapi.decompLZ4(data[0x56 : 0x56+cs], us)
    tail = data[0x56+cs:]

    # Data-driven: 0x48 non-zero => has skeleton (weight buffer present)
    has_skin = has_skeleton_flag(data) != 0

    vco = 0x34; ico = 0x38; vs = 0x60
    vnum = struct.unpack('<I', dr[vco:vco+4])[0]
    inum = struct.unpack('<I', dr[ico:ico+4])[0]
    vbuf_len = vnum * 16

    vbuf = dr[vs : vs+vbuf_len]
    gap = vbuf_len // 4
    us_start = vs + vbuf_len + gap
    uvbuf = dr[us_start : us_start+vbuf_len]

    idx_s = us_start + vbuf_len + (vnum * 8 if has_skin else 0)
    ibuf = dr[idx_s : idx_s + inum*4]

    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgBindUV1Buffer(uvbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgCommitTriangles(ibuf, noesis.RPGEODATA_UINT, inum, noesis.RPGEO_TRIANGLE)
    return parse_tail_bones(tail)

def parse_1E(data, filename):
    cs = struct.unpack('<I', data[0x4E:0x52])[0]
    us = struct.unpack('<I', data[0x52:0x56])[0]
    dr = rapi.decompLZ4(data[0x56 : 0x56+cs], us)
    tail = data[0x56+cs:]

    # Data-driven: 0x48 non-zero => has skeleton (weight buffer present)
    has_skin = has_skeleton_flag(data) != 0

    vnum = struct.unpack('<I', dr[0x74:0x78])[0]
    inum = struct.unpack('<I', dr[0x78:0x7C])[0] # index
    vs = 0xB3
    vbuf_len = vnum * 16
    vbuf = dr[vs : vs+vbuf_len]

    if has_skin:
        gap = vbuf_len // 4
        us_start = vs + vbuf_len + gap
        uvsz = vbuf_len
        idx_s = us_start + uvsz + vnum * 8
    else:
        gap = vnum * 4 - 4
        us_start = vs + vbuf_len + gap
        uvsz = vnum * 16
        idx_s = us_start + uvsz + 4

    uvbuf = dr[us_start : us_start+uvsz]
    ibuf = dr[idx_s : idx_s + inum*2]

    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, 16)
    # 1E uses 16-bit half-float UV; start offset is byte 4 within the 16-byte stride
    rapi.rpgBindUV1BufferOfs(uvbuf, noesis.RPGEODATA_HALFFLOAT, 16, 4)
    rapi.rpgCommitTriangles(ibuf, noesis.RPGEODATA_USHORT, inum, noesis.RPGEO_TRIANGLE)
    return parse_tail_bones(tail)

def parse_1F20(data, filename, version):
    if version == 0x1F:
        hdr = struct.unpack('<18IH3I', data[:86])
        bf = hdr[18]; csz = hdr[20]; usz = hdr[21]; cds = 86
    else:
        hdr = struct.unpack('<18IH4I', data[:90])
        bf = hdr[18]; csz = hdr[21]; usz = hdr[22]; cds = 90

    comp = data[cds : cds+csz]
    dr = rapi.decompLZ4(comp, usz)
    bds = cds + csz
    tail = data[bds:]

    # Data-driven skeleton flag: prefer container bf; matches 0x48 (verified identical)
    has_skin = (bf == 1) or (has_skeleton_flag(data) != 0)

    # Bone table sits after the compressed block (same layout as 1C/1E tail)
    bones = parse_tail_bones(tail) if has_skin else []

    # Standard mesh
    vnum = struct.unpack('<I', dr[116:120])[0]
    inum = struct.unpack('<I', dr[120:124])[0]
    vbs = 179

    vbuf = dr[vbs : vbs + vnum*16]
    uvbuf = dr[vbs + vnum*20 : vbs + vnum*36]

    if has_skin:
        wbuf = dr[vbs + vnum*36 : vbs + vnum*44]
        ibuf = dr[vbs + vnum*44 : vbs + vnum*44 + inum*2]
    else:
        wbuf = None
        ibuf = dr[vbs + vnum*36 : vbs + vnum*36 + inum*2]

    # Noesis binding
    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, 16)
    rapi.rpgBindUV1Buffer(uvbuf, noesis.RPGEODATA_HALFFLOAT, 16)
    rapi.rpgBindUV2BufferOfs(uvbuf, noesis.RPGEODATA_HALFFLOAT, 16, 4)

    if has_skin and wbuf and bones:
        bonemap = list(range(-1, len(bones)))
        bonemap[0] = 0
        rapi.rpgSetBoneMap(bonemap)
        rapi.rpgBindBoneIndexBuffer(wbuf, noesis.RPGEODATA_UBYTE, 8, 4)
        rapi.rpgBindBoneWeightBufferOfs(wbuf, noesis.RPGEODATA_UBYTE, 8, 4, 4)

    rapi.rpgCommitTriangles(ibuf, noesis.RPGEODATA_USHORT, inum, noesis.RPGEO_TRIANGLE)
    return bones

# ======================= ZipPos dequantization (new, correct) =======================
#
# The old logic treated the last vnum*4 bytes as raw coordinates, which is wrong.
# Verified across versions (1E/1F/20); the decompressed payload layout is:
#   0x00        4 bytes  leading field
#   0x04        AABB_a  (3f)  original bbox
#   0x10        AABB_b  (3f)
#   0x1c        AABB_a2 (3f)  dequant min
#   0x28        AABB_b2 (3f)  dequant max
#   0x34        quant_min (8f) UV etc. lower bound
#   0x54        quant_max (8f) UV etc. upper bound
#   0x74        shared_vertices (u32)
#   0x78        total_vertices  (u32) = total index count
#   0x7c        is_idx32        (u32)
#   0x80        num_points      (u32)
#   0x84..0x93  prop11..prop14  (4*u32)
#   0x94        load_norms(u8) load_info2(u8) pad(u8)
#   0x97        skip_mesh_pos(u32)  >0 => positions are compressed at the tail
#   0x9b        skip_uvs(u32)       >0 => UV is compressed
#   0x9f        flag3(u32)
#   0xa3        0x10 reserved
#   0xb3        mesh data region start
# Position: per vertex u32, 10-bit packed (qx<<20|qy<<10|qz), dequantized with AABB_a2/AABB_b2.
# UV: per vertex 4 bytes (u_hi,v_hi,u_lo,v_lo), 16-bit normalized then mapped by quant_min/max.

def _f32(d, o):
    return struct.unpack_from('<f', d, o)[0]

def _vec3(d, o):
    return (_f32(d, o), _f32(d, o+4), _f32(d, o+8))

def parse_zippos_payload(dr, has_skin, bones=None):
    """Parse decompressed ZipPos payload, dequantize position/UV and bind to Noesis."""
    p = 4  # leading field
    p += 12                       # AABB_a
    p += 12                       # AABB_b
    aabb_min = _vec3(dr, p); p += 12   # AABB_a2 dequant min
    aabb_max = _vec3(dr, p); p += 12   # AABB_b2 dequant max
    quant_min = [_f32(dr, p + i*4) for i in range(8)]; p += 32
    quant_max = [_f32(dr, p + i*4) for i in range(8)]; p += 32

    shared = _u32(dr, p); p += 4
    total = _u32(dr, p); p += 4
    is_idx32 = _u32(dr, p) != 0; p += 4
    num_points = _u32(dr, p); p += 4
    prop11 = _u32(dr, p); p += 4
    prop12 = _u32(dr, p); p += 4
    prop13 = _u32(dr, p); p += 4
    prop14 = _u32(dr, p); p += 4
    load_norms = dr[p] != 0; p += 1
    load_info2 = dr[p] != 0; p += 1
    p += 1
    skip_mesh_pos = _u32(dr, p); p += 4
    skip_uvs = _u32(dr, p); p += 4
    flag3 = _u32(dr, p); p += 4
    p += 0x10   # -> 0xB3

    face_count = total // 3
    idx_unit = 4 if is_idx32 else 2

    # read buffers in payload order (positions/normals/uv/weights/indices)
    inline_verts = b''
    if skip_mesh_pos == 0:
        inline_verts = dr[p : p + shared*16]
        p += shared * 16
    if load_norms:
        p += shared * 4
    # UV
    inline_uv_off = None
    if skip_uvs == 0:
        inline_uv_off = p
        p += shared * 16
    # inline (uncompressed) positions
    wbuf = None
    if has_skin:
        wbuf = dr[p : p + shared*8]
        p += shared * 8
    # inline UV
    ibuf = dr[p : p + face_count*3*idx_unit]
    p += face_count * 3 * idx_unit
    # skin weights
    if load_info2:
        p += total * idx_unit
    if num_points > 0:
        p += shared * idx_unit
    if prop11 > 0:
        p += shared * idx_unit
    if prop12 > 0:
        p += prop12 * idx_unit
    if prop13 > 0:
        p += prop13 * 4
    if prop14 > 0:
        p += prop14 * (8 if is_idx32 else 4)
    p += face_count * 4

    # --- positions ---
    if skip_mesh_pos > 0:
        ax, ay, az = aabb_min
        sx = aabb_max[0] - ax
        sy = aabb_max[1] - ay
        sz = aabb_max[2] - az
        vout = bytearray()
        for i in range(shared):
            pk = _u32(dr, p + i*4)
            qz = pk & 0x3FF
            qy = (pk >> 10) & 0x3FF
            qx = (pk >> 20) & 0x3FF
            vout += noePack('3f',
                            ax + (qx / 1023.0) * sx,
                            ay + (qy / 1023.0) * sy,
                            az + (qz / 1023.0) * sz)
        p += shared * 4
        p += shared  # extra 1 byte per vertex
        vbuf = bytes(vout)
        v_stride = 12
    else:
        vbuf = inline_verts
        v_stride = 16

    # --- UV ---
    uvbuf = None
    uv_stride = 16
    uv_type = noesis.RPGEODATA_HALFFLOAT
    uv_ofs = 4
    if skip_uvs > 0:
        umin, vmin = quant_min[0], quant_min[1]
        usz = quant_max[0] - umin
        vsz = quant_max[1] - vmin
        uvout = bytearray()
        for i in range(shared):
            off = p + i*4
            u_hi, v_hi, u_lo, v_lo = dr[off], dr[off+1], dr[off+2], dr[off+3]
            un = ((u_hi << 8) | u_lo) / 65535.0
            vn = ((v_hi << 8) | v_lo) / 65535.0
            uvout += noePack('2f', umin + un * usz, vmin + vn * vsz)
        p += shared * 4
        uvbuf = bytes(uvout)
        uv_stride = 8
        uv_type = noesis.RPGEODATA_FLOAT
        uv_ofs = 0
    elif inline_uv_off is not None:
        uvbuf = dr[inline_uv_off : inline_uv_off + shared*16]

    # --- bind ---
    rapi.rpgBindPositionBuffer(vbuf, noesis.RPGEODATA_FLOAT, v_stride)
    if uvbuf:
        if uv_ofs:
            rapi.rpgBindUV1BufferOfs(uvbuf, uv_type, uv_stride, uv_ofs)
        else:
            rapi.rpgBindUV1Buffer(uvbuf, uv_type, uv_stride)

    if has_skin and wbuf and bones:
        bonemap = list(range(-1, len(bones)))
        bonemap[0] = 0
        rapi.rpgSetBoneMap(bonemap)
        rapi.rpgBindBoneIndexBuffer(wbuf, noesis.RPGEODATA_UBYTE, 8, 4)
        rapi.rpgBindBoneWeightBufferOfs(wbuf, noesis.RPGEODATA_UBYTE, 8, 4, 4)

    idx_type = noesis.RPGEODATA_UINT if is_idx32 else noesis.RPGEODATA_USHORT
    rapi.rpgCommitTriangles(ibuf, idx_type, face_count*3, noesis.RPGEO_TRIANGLE)

# ======================= ZipPos loader (dispatched by filename) =======================
#
# ZipPos meshes are routed here by filename ('zippos' in the name).
# Decompression varies by header version; the decompressed payload then goes
# through parse_zippos_payload (10-bit position / 16-bit UV dequantization).
# Skeleton is detected purely from content (byte 0x48 / container flag), never
# from the filename.

def _zippos_decompress(data):
    """Return (payload, tail, has_skin) for a ZipPos .mesh, by header version.
    tail is the post-compression block that holds the bone table (if any)."""
    magic = data[:4]
    if magic in (b'\x1c\x00\x00\x00', b'\x1d\x00\x00\x00', b'\x1e\x00\x00\x00'):
        cs = struct.unpack('<I', data[0x4E:0x52])[0]
        us = struct.unpack('<I', data[0x52:0x56])[0]
        payload = rapi.decompLZ4(data[0x56 : 0x56+cs], us)
        tail = data[0x56+cs:]
        has_skin = has_skeleton_flag(data) != 0
        return payload, tail, has_skin
    if magic == b'\x1f\x00\x00\x00':
        hdr = struct.unpack('<18IH3I', data[:86])
        bf = hdr[18]; csz = hdr[20]; usz = hdr[21]; cds = 86
    elif magic == b'\x20\x00\x00\x00':
        hdr = struct.unpack('<18IH4I', data[:90])
        bf = hdr[18]; csz = hdr[21]; usz = hdr[22]; cds = 90
    else:
        raise ValueError("ZipPos: unsupported header " + binascii.hexlify(magic).decode('ascii'))
    payload = rapi.decompLZ4(data[cds : cds+csz], usz)
    tail = data[cds+csz:]
    has_skin = (bf == 1) or (has_skeleton_flag(data) != 0)
    return payload, tail, has_skin

def noepyLoadZipModel(data, mdlList):
    rapi.rpgCreateContext()
    try:
        payload, tail, has_skin = _zippos_decompress(data)
        bones = parse_tail_bones(tail) if has_skin else []
        parse_zippos_payload(payload, has_skin, bones)
    except Exception as e:
        print("Error parsing ZipPos mesh: " + str(e))
        return 0

    try:
        mdl = rapi.rpgConstructModel()
    except:
        mdl = NoeModel()

    mdl.setModelMaterials(NoeModelMaterials([], [NoeMaterial('default','')]))
    if bones:
        mdl.setBones(bones)

    mdlList.append(mdl)
    return 1