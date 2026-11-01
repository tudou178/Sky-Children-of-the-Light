#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import struct
import shutil
from typing import List, Dict, Any, Tuple, Optional

# ----------------------------------------------------------------------
# 常量定义
# ----------------------------------------------------------------------
NORMAL, STRING, OBJECT_PTR, ARRAY = 0, 1, 2, 3

# ----------------------------------------------------------------------
# 原始解析器（用于读取 .bin 文件）
# ----------------------------------------------------------------------
class TGCLParser:
    def __init__(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.buf = f.read()
        self.filepath = filepath
        self.header = None
        self.types = []          # 类型表
        self.mem_vars = []       # 成员变量表
        self.objects = []        # 解析出的对象
        self.obj_field_offsets = {}   # { obj_index: { field_name: (offset, mem_var) } }

    def parse_header(self):
        if self.buf[:4] != b'TGCL':
            raise ValueError(f"Invalid magic: {self.buf[:4]}, expected TGCL")

        self.header = {
            'magic': self.buf[0:4].decode('ascii'),
            'version': struct.unpack('<I', self.buf[4:8])[0],
            'numTypes': struct.unpack('<I', self.buf[8:12])[0],
            'numMemVars': struct.unpack('<I', self.buf[12:16])[0],
            'numObjects': struct.unpack('<I', self.buf[16:20])[0],
            'numRefs': struct.unpack('<I', self.buf[20:24])[0],
            'typesOffset': struct.unpack('<I', self.buf[24:28])[0],
            'memVarsOffset': struct.unpack('<I', self.buf[28:32])[0],
            'dataOffset': struct.unpack('<I', self.buf[32:36])[0],
            'stringsOffset': struct.unpack('<I', self.buf[36:40])[0],
            'fileSize': struct.unpack('<I', self.buf[40:44])[0]
        }

    def read_name_string(self, offset: int) -> str:
        start = self.header['dataOffset'] + offset
        end = start
        while end < len(self.buf) and self.buf[end] != 0:
            end += 1
        return self.buf[start:end].decode('utf-8')

    def parse_types(self):
        off = self.header['typesOffset']
        for i in range(self.header['numTypes']):
            name_offset = struct.unpack('<I', self.buf[off:off+4])[0]
            first_mem_var = struct.unpack('<I', self.buf[off+4:off+8])[0]
            num_mem_vars = struct.unpack('<I', self.buf[off+8:off+12])[0]
            self.types.append({
                'index': i,
                'name': self.read_name_string(name_offset),
                'firstMemVar': first_mem_var,
                'numMemVars': num_mem_vars
            })
            off += 12

    def parse_mem_vars(self):
        off = self.header['memVarsOffset']
        for i in range(self.header['numMemVars']):
            var_type = struct.unpack('<I', self.buf[off:off+4])[0]
            name_offset = struct.unpack('<I', self.buf[off+4:off+8])[0]
            size = struct.unpack('<I', self.buf[off+8:off+12])[0]
            extra = struct.unpack('<I', self.buf[off+12:off+16])[0]
            self.mem_vars.append({
                'index': i,
                'varType': var_type,
                'name': self.read_name_string(name_offset),
                'size': size,
                'extra': extra
            })
            off += 16

    def get_type_mem_vars(self, type_index: int) -> List[Dict]:
        t = self.types[type_index]
        start = t['firstMemVar']
        end = start + t['numMemVars']
        return self.mem_vars[start:end]

    def read_value(self, pos: int, mem_var: Dict) -> Tuple[Any, int]:
        var_type = mem_var['varType']
        size = mem_var['size']
        name = mem_var['name']
        extra = mem_var['extra']

        if var_type == STRING:
            end = pos
            while end < len(self.buf) and self.buf[end] != 0:
                end += 1
            value = self.buf[pos:end].decode('utf-8')
            return value, end - pos + 1
        elif var_type == OBJECT_PTR:
            idx = struct.unpack('<I', self.buf[pos:pos+4])[0]
            if idx == 0xFFFFFFFF:
                value = None
            else:
                value = f"@ref_{idx}"
            return value, 4
        elif var_type == ARRAY:
            count = struct.unpack('<I', self.buf[pos:pos+4])[0]
            array_pos = pos + 4
            if extra == 0xFFFFFFFF:
                refs = []
                for _ in range(count):
                    idx = struct.unpack('<I', self.buf[array_pos:array_pos+4])[0]
                    if idx == 0xFFFFFFFF:
                        refs.append(None)
                    else:
                        refs.append(f"@ref_{idx}")
                    array_pos += 4
                return refs, array_pos - pos
            elif extra < len(self.types):
                elem_mem_vars = self.get_type_mem_vars(extra)
                elements = []
                for _ in range(count):
                    elem = {}
                    for emv in elem_mem_vars:
                        val, adv = self.read_value(array_pos, emv)
                        elem[emv['name']] = val
                        array_pos += adv
                    elements.append(elem)
                return elements, array_pos - pos
            else:
                elem_size = extra if extra != 0 else 4
                data = []
                for _ in range(count):
                    data.append(self.buf[array_pos:array_pos+elem_size].hex())
                    array_pos += elem_size
                return data, array_pos - pos
        else:  # NORMAL
            return self.read_normal_value(pos, size, name, mem_var)

    def read_normal_value(self, pos: int, size: int, name: str, mem_var: Dict) -> Tuple[Any, int]:
        if size == 1:
            v = self.buf[pos]
            if v in (0, 1):
                return bool(v), size
            return v, size
        elif size == 4:
            if 'Guid' in name or 'guid' in name:
                return struct.unpack('<I', self.buf[pos:pos+4])[0], size
            int_val = struct.unpack('<i', self.buf[pos:pos+4])[0]
            float_val = struct.unpack('<f', self.buf[pos:pos+4])[0]
            if (abs(float_val) < 1e10 and abs(float_val) > 1e-10) or float_val == 0.0:
                return float_val, size
            else:
                return int_val, size
        elif size == 8:
            return struct.unpack('<d', self.buf[pos:pos+8])[0], size
        elif size == 12:
            return {
                'x': struct.unpack('<f', self.buf[pos:pos+4])[0],
                'y': struct.unpack('<f', self.buf[pos+4:pos+8])[0],
                'z': struct.unpack('<f', self.buf[pos+8:pos+12])[0]
            }, size
        elif size == 16:
            return {
                'x': struct.unpack('<f', self.buf[pos:pos+4])[0],
                'y': struct.unpack('<f', self.buf[pos+4:pos+8])[0],
                'z': struct.unpack('<f', self.buf[pos+8:pos+12])[0],
                'w': struct.unpack('<f', self.buf[pos+12:pos+16])[0]
            }, size
        elif size == 64:
            floats = struct.unpack('<16f', self.buf[pos:pos+64])
            return [list(floats[i*4:(i+1)*4]) for i in range(4)], size
        else:
            return self.buf[pos:pos+size].hex(), size

    def parse_objects(self):
        pos = self.header['stringsOffset']
        for i in range(self.header['numObjects']):
            if pos >= len(self.buf):
                break
            obj, pos, offsets = self.parse_object(pos, i)
            self.objects.append(obj)
            self.obj_field_offsets[i] = offsets

    def parse_object(self, pos: int, idx: int) -> Tuple[Dict, int, Dict]:
        start_pos = pos
        type_index = struct.unpack('<I', self.buf[pos:pos+4])[0]
        pos += 4
        name_end = pos
        while name_end < len(self.buf) and self.buf[name_end] != 0:
            name_end += 1
        name = self.buf[pos:name_end].decode('utf-8')
        pos = name_end + 1

        obj = {
            '_index': idx,
            '_type': self.types[type_index]['name'] if type_index < len(self.types) else f"unknown_type_{type_index}",
            '_typeIndex': type_index,
            '_name': name,
            '_startPos': start_pos
        }
        field_offsets = {}

        if type_index < len(self.types):
            mem_vars = self.get_type_mem_vars(type_index)
            for mv in mem_vars:
                field_offsets[mv['name']] = (pos, mv)
                try:
                    val, adv = self.read_value(pos, mv)
                    obj[mv['name']] = val
                    pos += adv
                except Exception as e:
                    print(f"Warning: error reading {obj['_type']}.{mv['name']}: {e}")
                    pos += mv.get('size', 4)
        obj['_nextPos'] = pos
        return obj, pos, field_offsets

    def parse(self):
        self.parse_header()
        self.parse_types()
        self.parse_mem_vars()
        self.parse_objects()
        return self

# ----------------------------------------------------------------------
# JSON 编码辅助函数（兼容 Objects.level.bin(1).json 格式）
# ----------------------------------------------------------------------
def format_value_for_json(val, idx_to_key):
    if isinstance(val, bool):
        return val
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, float):
        return format(val, '.6f')
    elif isinstance(val, dict):
        if set(val.keys()) == {'x', 'y', 'z', 'w'}:
            return [format(val['x'], '.6f'), format(val['y'], '.6f'),
                    format(val['z'], '.6f'), format(val['w'], '.6f')]
        elif set(val.keys()) == {'x', 'y', 'z'}:
            return [format(val['x'], '.6f'), format(val['y'], '.6f'),
                    format(val['z'], '.6f')]
        else:
            return {k: format_value_for_json(v, idx_to_key) for k, v in val.items()}
    elif isinstance(val, list):
        return [format_value_for_json(item, idx_to_key) for item in val]
    elif isinstance(val, str):
        if val.startswith('@ref_'):
            idx_str = val.split('_')[1]
            try:
                idx = int(idx_str)
                if idx in idx_to_key:
                    return idx_to_key[idx]
            except ValueError:
                pass
            return f'BstNode_{idx_str}'
        elif val.startswith('@object_'):
            idx = val.split('_')[1]
            return f'BstNode_{idx}'
        else:
            return val
    else:
        return str(val)

def build_json_compatible_dict(parser):
    classes = {}
    for type_info in parser.types:
        class_name = type_info['name']
        mem_vars = parser.get_type_mem_vars(type_info['index'])
        class_def = {}
        for mv in mem_vars:
            prop = {
                'propertyType': mv['varType'],
                'objectByteSize': mv['size'],
                'arrayIndex': mv['extra']
            }
            class_def[mv['name']] = prop
        classes[class_name] = class_def

    idx_to_key = {}
    objects = {}
    for obj in parser.objects:
        obj_idx = obj['_index']
        obj_name = obj.get('_name', '')
        if obj_name and isinstance(obj_name, str) and obj_name.strip():
            obj_key = obj_name
        else:
            obj_key = f'BstNode_{obj_idx}'
        idx_to_key[obj_idx] = obj_key

    for obj in parser.objects:
        obj_idx = obj['_index']
        obj_key = idx_to_key[obj_idx]
        class_name = obj['_type']
        fields = {}
        for mv in parser.get_type_mem_vars(obj['_typeIndex']):
            field_name = mv['name']
            if field_name in obj:
                raw_val = obj[field_name]
                fields[field_name] = format_value_for_json(raw_val, idx_to_key)
            else:
                fields[field_name] = None
        objects[obj_key] = {class_name: fields}

    if parser.header is None:
        version = 1
        memory_size = str(len(parser.buf))
    else:
        version = parser.header.get('version', 1)
        memory_size = str(parser.header.get('fileSize', len(parser.buf)))
    root = {
        'version': version,
        'MemorySize': memory_size,
        'classes': classes,
        **objects
    }
    return root

# ----------------------------------------------------------------------
# 序列化辅助函数（支持 ARRAY 写回，且兼容 JSON 中的字符串数字）
# ----------------------------------------------------------------------
def serialize_field_value(value: Any, mem_var: Dict, types: List[Dict], mem_vars_list: List[Dict], key_to_idx: Dict) -> bytes:
    var_type = mem_var['varType']
    size = mem_var['size']
    extra = mem_var['extra']
    name = mem_var['name']

    def to_float(v):
        if isinstance(v, str):
            return float(v)
        return float(v)

    if var_type == NORMAL:
        if size == 1:
            if isinstance(value, bool):
                return bytes([1 if value else 0])
            elif isinstance(value, str):
                if value.lower() in ('true', '1'):
                    return b'\x01'
                elif value.lower() in ('false', '0'):
                    return b'\x00'
                else:
                    return bytes([int(float(value)) & 0xFF])
            else:
                return bytes([int(value) & 0xFF])
        elif size == 4:
            if 'Guid' in name or 'guid' in name:
                if isinstance(value, str):
                    val = int(value)
                else:
                    val = int(value)
                return struct.pack('<I', val & 0xFFFFFFFF)

            if isinstance(value, bool):
                return struct.pack('<i', 1 if value else 0)
            elif isinstance(value, str):
                try:
                    if '.' in value or 'e' in value.lower():
                        return struct.pack('<f', float(value))
                    else:
                        ival = int(value)
                        if ival < -2147483648 or ival > 2147483647:
                            return struct.pack('<I', ival & 0xFFFFFFFF)
                        return struct.pack('<i', ival)
                except ValueError:
                    return struct.pack('<i', 0)
            elif isinstance(value, float):
                return struct.pack('<f', value)
            elif isinstance(value, int):
                if value < -2147483648 or value > 2147483647:
                    return struct.pack('<I', value & 0xFFFFFFFF)
                return struct.pack('<i', value)
            else:
                return struct.pack('<i', 0)
        elif size == 8:
            return struct.pack('<d', to_float(value))
        elif size == 12:
            if isinstance(value, dict):
                x = to_float(value.get('x', 0.0))
                y = to_float(value.get('y', 0.0))
                z = to_float(value.get('z', 0.0))
            elif isinstance(value, (list, tuple)):
                x = to_float(value[0])
                y = to_float(value[1])
                z = to_float(value[2])
            else:
                raise ValueError(f"无法将 {value} 转换为 Vector3")
            return struct.pack('<fff', x, y, z)
        elif size == 16:
            if isinstance(value, dict):
                x = to_float(value.get('x', 0.0))
                y = to_float(value.get('y', 0.0))
                z = to_float(value.get('z', 0.0))
                w = to_float(value.get('w', 0.0))
            elif isinstance(value, (list, tuple)):
                x = to_float(value[0])
                y = to_float(value[1])
                z = to_float(value[2])
                w = to_float(value[3])
            else:
                raise ValueError(f"无法将 {value} 转换为 Vector4")
            return struct.pack('<ffff', x, y, z, w)
        elif size == 64:
            flat = []
            if isinstance(value, list) and len(value) == 4:
                for row in value:
                    if isinstance(row, (list, tuple)):
                        for elem in row[:4]:
                            flat.append(to_float(elem))
                    else:
                        flat.extend([0.0] * 4)
            else:
                flat = [0.0] * 16
            return struct.pack('<16f', *flat)
        else:
            if isinstance(value, str) and len(value) == size * 2:
                return bytes.fromhex(value)
            elif isinstance(value, bytes):
                return value[:size].ljust(size, b'\x00')
            else:
                return str(value).encode('utf-8')[:size].ljust(size, b'\x00')

    elif var_type == STRING:
        return str(value).encode('utf-8') + b'\x00'

    elif var_type == OBJECT_PTR:
        if value is None:
            idx = 0xFFFFFFFF
        elif isinstance(value, int):
            idx = value
        elif isinstance(value, str):
            if value.startswith('BstNode_'):
                idx = key_to_idx.get(value, 0xFFFFFFFF)
            elif value.startswith('@ref_'):
                idx = int(value.split('_')[1])
            elif value.startswith('@object_'):
                idx = int(value.split('_')[1])
            else:
                idx = key_to_idx.get(value, 0xFFFFFFFF)
        else:
            idx = 0xFFFFFFFF
        return struct.pack('<I', idx & 0xFFFFFFFF)

    elif var_type == ARRAY:
        if not isinstance(value, list):
            raise ValueError("ARRAY 字段的值必须是列表")
        count = len(value)
        buf = bytearray()
        buf += struct.pack('<I', count)

        if extra == 0xFFFFFFFF:  # 对象引用数组
            for item in value:
                if item is None:
                    idx = 0xFFFFFFFF
                elif isinstance(item, int):
                    idx = item
                elif isinstance(item, str):
                    if item.startswith('BstNode_'):
                        idx = key_to_idx.get(item, 0xFFFFFFFF)
                    elif item.startswith('@ref_'):
                        idx = int(item.split('_')[1])
                    elif item.startswith('@object_'):
                        idx = int(item.split('_')[1])
                    else:
                        idx = key_to_idx.get(item, 0xFFFFFFFF)
                else:
                    idx = 0xFFFFFFFF
                buf += struct.pack('<I', idx & 0xFFFFFFFF)
        elif extra < len(types):
            # 复杂元素数组，需要获取元素类型的成员变量定义
            elem_mem_vars = []
            for t in types:
                if t['index'] == extra:
                    start = t['firstMemVar']
                    end = start + t['numMemVars']
                    elem_mem_vars = mem_vars_list[start:end]
                    break
            if not elem_mem_vars:
                raise ValueError(f"无法找到数组元素类型 (index={extra})")
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("复杂数组元素必须是字典")
                for emv in elem_mem_vars:
                    fname = emv['name']
                    if fname not in item:
                        raise KeyError(f"数组元素缺失字段: {fname}")
                    sub_val = item[fname]
                    field_bytes = serialize_field_value(sub_val, emv, types, mem_vars_list, key_to_idx)
                    buf += field_bytes
        else:
            elem_size = extra if extra != 0 else 4
            for item in value:
                if isinstance(item, str):
                    try:
                        data = bytes.fromhex(item)
                    except ValueError:
                        data = b'\x00' * elem_size
                    if len(data) != elem_size:
                        data = data[:elem_size].ljust(elem_size, b'\x00')
                elif isinstance(item, int):
                    if elem_size == 1:
                        data = struct.pack('<B', item & 0xFF)
                    elif elem_size == 2:
                        data = struct.pack('<H', item & 0xFFFF)
                    elif elem_size == 4:
                        data = struct.pack('<I', item & 0xFFFFFFFF)
                    elif elem_size == 8:
                        data = struct.pack('<Q', item & 0xFFFFFFFFFFFFFFFF)
                    else:
                        data = struct.pack('<I', item)[:elem_size].ljust(elem_size, b'\x00')
                elif isinstance(item, bytes):
                    data = item[:elem_size].ljust(elem_size, b'\x00')
                else:
                    data = bytes(str(item), 'utf-8')[:elem_size].ljust(elem_size, b'\x00')
                buf += data
        return bytes(buf)

    else:
        raise ValueError(f"不支持的 var_type: {var_type}")

def serialize_object(obj: Dict, type_index: int, mem_vars_for_type: List[Dict],
                     types: List[Dict], all_mem_vars: List[Dict],
                     key_to_idx: Dict) -> bytes:
    """将对象序列化为字节串"""
    buf = bytearray()
    buf += struct.pack('<I', type_index)
    name = obj.get('_name', '')
    buf += name.encode('utf-8') + b'\x00'
    for mv in mem_vars_for_type:
        field_name = mv['name']
        if field_name not in obj:
            raise KeyError(f"对象缺失必要字段: {field_name}")
        val = obj[field_name]
        field_bytes = serialize_field_value(val, mv, types, all_mem_vars, key_to_idx)
        buf += field_bytes
    return bytes(buf)

# ----------------------------------------------------------------------
# 辅助函数：递归统计对象中的引用数量
# ----------------------------------------------------------------------
def count_refs_in_obj(obj: Dict, mem_vars: List[Dict], types: List[Dict], mem_vars_list: List[Dict]) -> int:
    cnt = 0
    for mv in mem_vars:
        val = obj.get(mv['name'])
        if mv['varType'] == OBJECT_PTR:
            if val is not None:
                cnt += 1
        elif mv['varType'] == ARRAY:
            if isinstance(val, list):
                if mv['extra'] == 0xFFFFFFFF:
                    cnt += sum(1 for item in val if item is not None)
                elif mv['extra'] < len(types):
                    # 复杂元素数组，递归统计内部引用
                    elem_mem_vars = []
                    for t in types:
                        if t['index'] == mv['extra']:
                            start = t['firstMemVar']
                            end = start + t['numMemVars']
                            elem_mem_vars = mem_vars_list[start:end]
                            break
                    for item in val:
                        if isinstance(item, dict):
                            cnt += count_refs_in_obj(item, elem_mem_vars, types, mem_vars_list)
    return cnt

# ----------------------------------------------------------------------
# JSON 转 bin 核心功能（纯 JSON 驱动，不依赖模板 bin）
# ----------------------------------------------------------------------
def json_to_bin(json_path: str, output_bin_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    if 'classes' not in json_data:
        raise ValueError("JSON 中缺少 'classes' 字段")

    json_classes = json_data['classes']

    # 提取对象列表（根级非元数据键）
    objects = []
    for key, value in json_data.items():
        if key in ('version', 'MemorySize', 'classes'):
            continue
        if not isinstance(value, dict):
            continue
        if len(value) != 1:
            continue
        class_name = next(iter(value))
        fields = value[class_name]
        obj = {'_name': key, '_type': class_name}
        obj.update(fields)
        objects.append(obj)

    # 分配对象索引（按提取顺序，即 JSON 中的出现顺序）
    for idx, obj in enumerate(objects):
        obj['_index'] = idx

    # 建立对象名到索引的映射（用于引用解析）
    key_to_idx = {}
    for obj in objects:
        name = obj.get('_name', '')
        if name and isinstance(name, str) and name.strip():
            key = name
        else:
            key = f'BstNode_{obj["_index"]}'
        key_to_idx[key] = obj['_index']

    # 构建类型索引映射（从0开始连续分配，按 classes 在 JSON 中的顺序）
    type_index_map = {}
    for i, type_name in enumerate(json_classes.keys()):
        type_index_map[type_name] = i

    # 构建类型表与成员变量表，同时构建字符串池（去重）
    new_types = []
    new_mem_vars = []
    string_pool = bytearray()
    string_offsets = {}

    def add_string(s: str) -> int:
        if s in string_offsets:
            return string_offsets[s]
        offset = len(string_pool)
        string_pool.extend(s.encode('utf-8') + b'\x00')
        string_offsets[s] = offset
        return offset

    for type_name, type_idx in sorted(type_index_map.items(), key=lambda x: x[1]):
        fields_def = json_classes[type_name]

        first_mem_var = len(new_mem_vars)
        num_mem_vars = 0

        type_name_offset = add_string(type_name)

        # 按 JSON 中定义的字段顺序（Python 3.7+ 保持插入顺序）
        for field_name, field_def in fields_def.items():
            var_type = field_def['propertyType']
            size = field_def['objectByteSize']
            extra = field_def['arrayIndex']
            field_name_offset = add_string(field_name)
            new_mem_vars.append({
                'index': len(new_mem_vars),
                'varType': var_type,
                'name': field_name,
                'size': size,
                'extra': extra,
                'name_offset': field_name_offset
            })
            num_mem_vars += 1

        new_types.append({
            'index': type_idx,
            'name': type_name,
            'firstMemVar': first_mem_var,
            'numMemVars': num_mem_vars,
            'name_offset': type_name_offset
        })

    # 确保 new_types 按索引排序
    new_types.sort(key=lambda t: t['index'])

    string_pool_bytes = bytes(string_pool)

    # 构建类型名到索引的快速映射
    type_name_to_index = {t['name']: t['index'] for t in new_types}

    # 精确统计 numRefs（递归统计所有对象中的对象引用数量）
    num_refs = 0
    for obj in objects:
        class_name = obj['_type']
        type_idx = type_name_to_index[class_name]
        t = new_types[type_idx]
        obj_mem_vars = new_mem_vars[t['firstMemVar']:t['firstMemVar']+t['numMemVars']]
        num_refs += count_refs_in_obj(obj, obj_mem_vars, new_types, new_mem_vars)

    # 序列化所有对象
    objects_bytes = bytearray()
    for obj in objects:
        class_name = obj['_type']
        if class_name not in type_name_to_index:
            raise ValueError(f"未知类型名: {class_name}")
        type_index = type_name_to_index[class_name]
        t = new_types[type_index]
        obj_mem_vars = new_mem_vars[t['firstMemVar']:t['firstMemVar']+t['numMemVars']]
        obj_bytes = serialize_object(obj, type_index, obj_mem_vars, new_types, new_mem_vars, key_to_idx)
        objects_bytes += obj_bytes

    new_num_objects = len(objects)

    # 计算各区域大小和偏移
    header_size = 44
    types_size = len(new_types) * 12
    mem_vars_size = len(new_mem_vars) * 16

    types_offset = header_size
    mem_vars_offset = types_offset + types_size
    data_offset = mem_vars_offset + mem_vars_size
    strings_offset = data_offset + len(string_pool_bytes)
    file_size = strings_offset + len(objects_bytes)

    # 构建头部
    version = json_data.get('version', 1)
    header = bytearray(44)
    header[0:4] = b'TGCL'
    struct.pack_into('<I', header, 4, version)
    struct.pack_into('<I', header, 8, len(new_types))
    struct.pack_into('<I', header, 12, len(new_mem_vars))
    struct.pack_into('<I', header, 16, new_num_objects)
    struct.pack_into('<I', header, 20, num_refs)
    struct.pack_into('<I', header, 24, types_offset)
    struct.pack_into('<I', header, 28, mem_vars_offset)
    struct.pack_into('<I', header, 32, data_offset)
    struct.pack_into('<I', header, 36, strings_offset)
    struct.pack_into('<I', header, 40, file_size)

    # 写入类型表
    types_buf = bytearray()
    for t in new_types:
        types_buf.extend(struct.pack('<I', t['name_offset']))
        types_buf.extend(struct.pack('<I', t['firstMemVar']))
        types_buf.extend(struct.pack('<I', t['numMemVars']))

    # 写入成员变量表
    mem_vars_buf = bytearray()
    for mv in new_mem_vars:
        mem_vars_buf.extend(struct.pack('<I', mv['varType']))
        mem_vars_buf.extend(struct.pack('<I', mv['name_offset']))
        mem_vars_buf.extend(struct.pack('<I', mv['size']))
        mem_vars_buf.extend(struct.pack('<I', mv['extra']))

    # 合并文件内容
    new_content = header + types_buf + mem_vars_buf + string_pool_bytes + objects_bytes

    with open(output_bin_path, 'wb') as f:
        f.write(new_content)

    print(f"成功生成 bin 文件: {output_bin_path}")
    print(f"对象数量: {new_num_objects}, 文件大小: {file_size} 字节, 引用数: {num_refs}")

# ----------------------------------------------------------------------
# 菜单界面
# ----------------------------------------------------------------------
def parse_bin_file():
    path = input("请输入 .bin 文件路径: ").strip()
    if not path or not os.path.isfile(path):
        print("文件不存在或路径无效。")
        return
    try:
        parser = TGCLParser(path)
        parser.parse()
        out_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(out_path):
            bak_path = out_path + ".bak"
            shutil.copy2(out_path, bak_path)
            print(f"已备份原 JSON 至 {bak_path}")
        result = build_json_compatible_dict(parser)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"解析完成，JSON 保存至 {out_path}")
        print(f"对象数量: {len(parser.objects)}")
    except Exception as e:
        print(f"解析失败: {e}")

def json_to_bin_menu():
    json_path = input("请输入 JSON 文件路径: ").strip()
    if not json_path or not os.path.isfile(json_path):
        print("JSON 文件不存在。")
        return
    output_bin = input("请输入输出 .bin 文件路径: ").strip()
    if not output_bin:
        print("输出路径不能为空。")
        return
    try:
        json_to_bin(json_path, output_bin)
    except Exception as e:
        print(f"转换失败: {e}")

def main_menu():
    while True:
        print("\n===== TGCL 编辑器 =====")
        print("1. 解析 .bin 为 JSON（自动备份）")
        print("2. JSON 转 bin（纯 JSON 驱动）")
        print("3. 退出")
        choice = input("请选择: ").strip()
        if choice == '1':
            parse_bin_file()
        elif choice == '2':
            json_to_bin_menu()
        elif choice == '3':
            break
        else:
            print("无效选项，请重试。")

# ----------------------------------------------------------------------
# 测试函数：验证 JSON 往返语义一致性，而非二进制完全一致
# ----------------------------------------------------------------------
def test_roundtrip(bin_path: str):
    """测试 round-trip 语义一致性：将 bin 解析为 JSON，再用 JSON 生成 bin，最后重新解析为 JSON，比较两个 JSON 是否等价。"""
    import tempfile, os, json
    print(f"测试文件: {bin_path}")
    # 解析为 JSON
    parser = TGCLParser(bin_path)
    parser.parse()
    json_data1 = build_json_compatible_dict(parser)

    # 保存临时 JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data1, f, indent=2, ensure_ascii=False)
        json_path = f.name

    # 转换回 bin
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        bin_out_path = f.name

    try:
        json_to_bin(json_path, bin_out_path)

        # 重新解析生成的 bin
        parser2 = TGCLParser(bin_out_path)
        parser2.parse()
        json_data2 = build_json_compatible_dict(parser2)

        # 比较两个 JSON 是否深度相等
        if json_data1 == json_data2:
            print("✓ JSON 语义完全一致，往返转换成功！")
        else:
            print("错误：JSON 不一致，可能存在逻辑缺陷。")
            # 可选：输出差异
            from deepdiff import DeepDiff
            diff = DeepDiff(json_data1, json_data2, ignore_order=True)
            print(diff)
    except ImportError:
        # 如果没有 deepdiff，简单比较
        if json_data1 == json_data2:
            print("✓ JSON 语义完全一致，往返转换成功！")
        else:
            print("错误：JSON 不一致，请检查。")
    finally:
        os.unlink(json_path)
        os.unlink(bin_out_path)

def test_modification(bin_path: str):
    """测试修改字段值并验证序列化"""
    import tempfile, os, json
    print(f"测试修改: {bin_path}")
    parser = TGCLParser(bin_path)
    parser.parse()
    json_data = build_json_compatible_dict(parser)
    obj_key = None
    for k, v in json_data.items():
        if k in ('version', 'MemorySize', 'classes'):
            continue
        obj_key = k
        break
    if obj_key is None:
        print("没有找到对象，跳过修改测试")
        return
    obj = json_data[obj_key]
    class_name = next(iter(obj))
    fields = obj[class_name]
    target_field = None
    for fname, fval in fields.items():
        if isinstance(fval, (int, str)) and (isinstance(fval, str) and fval.isdigit() or isinstance(fval, int)):
            target_field = fname
            break
    if target_field is None:
        print("未找到合适的字段，跳过修改测试")
        return
    original_value = fields[target_field]
    print(f"修改对象 {obj_key} 字段 {target_field}: {original_value} -> 999")
    fields[target_field] = 999
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        json_path = f.name
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        bin_out_path = f.name
    try:
        json_to_bin(json_path, bin_out_path)
        parser2 = TGCLParser(bin_out_path)
        parser2.parse()
        json_data2 = build_json_compatible_dict(parser2)
        obj2 = json_data2.get(obj_key)
        if obj2 is None:
            print("错误：修改后的 bin 中找不到原对象")
            return
        fields2 = obj2[class_name]
        new_value = fields2.get(target_field)
        if new_value == 999:
            print("✓ 修改成功并验证")
        else:
            print(f"错误：字段值未更新，当前值 {new_value}")
    finally:
        os.unlink(json_path)
        os.unlink(bin_out_path)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        if len(sys.argv) > 2:
            test_roundtrip(sys.argv[2])
        else:
            test_roundtrip('/mnt/sdcard/.system_ext/Objects.level.bin')
    elif len(sys.argv) > 1 and sys.argv[1] == 'testmod':
        if len(sys.argv) > 2:
            test_modification(sys.argv[2])
        else:
            test_modification('/mnt/sdcard/.system_ext/Objects.level.bin')
    else:
        main_menu()