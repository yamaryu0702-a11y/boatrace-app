"""最小限の純Python LZH(-lh5-/-lh0-)展開。ボートレース公式データ用。"""
import struct


class _Bits:
    def __init__(self, data):
        self.d = data
        self.pos = 0  # bit position

    def get(self, n):
        if n == 0:
            return 0
        i = self.pos >> 3
        chunk = int.from_bytes(self.d[i:i + 4].ljust(4, b"\0"), "big")
        v = (chunk >> (32 - (self.pos & 7) - n)) & ((1 << n) - 1)
        self.pos += n
        return v


def _build(lengths):
    """canonical huffman -> {(len, code): sym}"""
    table = {}
    code = 0
    maxlen = max(lengths) if lengths else 0
    for ln in range(1, maxlen + 1):
        for sym, l in enumerate(lengths):
            if l == ln:
                table[(ln, code)] = sym
                code += 1
        code <<= 1
    return table


def _dec(bits, table, single):
    if single is not None:
        return single
    code = 0
    for ln in range(1, 20):
        code = (code << 1) | bits.get(1)
        s = table.get((ln, code))
        if s is not None:
            return s
    raise ValueError("bad huffman code")


def _read_pt(bits, nn, nbit, special):
    n = bits.get(nbit)
    if n == 0:
        return [0] * nn, bits.get(nbit)
    lens = [0] * nn
    i = 0
    while i < n:
        c = bits.get(3)
        if c == 7:
            while bits.get(1) == 1:
                c += 1
        lens[i] = c
        i += 1
        if i == special:
            i += bits.get(2)
    return lens, None


def _read_c(bits, t_table, t_single):
    n = bits.get(9)
    if n == 0:
        return [0] * 510, bits.get(9)
    lens = [0] * 510
    i = 0
    while i < n:
        c = _dec(bits, t_table, t_single)
        if c <= 2:
            if c == 0:
                c = 1
            elif c == 1:
                c = bits.get(4) + 3
            else:
                c = bits.get(9) + 20
            i += c
        else:
            lens[i] = c - 2
            i += 1
    return lens, None


def _lh5(comp, origsize):
    bits = _Bits(comp)
    out = bytearray()
    blocksize = 0
    while len(out) < origsize:
        if blocksize == 0:
            blocksize = bits.get(16)
            tl, ts = _read_pt(bits, 19, 5, 3)
            tt = _build(tl)
            cl, cs = _read_c(bits, tt, ts)
            ct = _build(cl)
            pl, ps = _read_pt(bits, 14, 4, -1)
            pt = _build(pl)
        blocksize -= 1
        c = _dec(bits, ct, cs)
        if c < 256:
            out.append(c)
        else:
            length = c - 256 + 3
            j = _dec(bits, pt, ps)
            if j != 0:
                j = (1 << (j - 1)) + bits.get(j - 1)
            start = len(out) - j - 1
            for k in range(length):
                out.append(out[start + k])
    return bytes(out[:origsize])


def extract(data):
    """LZHアーカイブ(バイト列)から [(名前, 内容bytes), ...] を返す。"""
    files = []
    p = 0
    while p < len(data) and data[p] != 0:
        hsize = data[p]
        method = data[p + 2:p + 7].decode("ascii")
        csize, osize = struct.unpack_from("<II", data, p + 7)
        level = data[p + 20]
        if level == 2:
            hsize2 = struct.unpack_from("<H", data, p)[0]
            name = ""
            q = p + 24  # 拡張ヘッダ先頭(level2)
            while True:
                nxt = struct.unpack_from("<H", data, q)[0]
                if nxt == 0:
                    break
                if data[q + 2] == 1:
                    name = data[q + 3:q + nxt - 0].decode("cp932", "replace")
                q += nxt
            body = p + hsize2
        elif level == 1:
            nl = data[p + 21]
            name = data[p + 22:p + 22 + nl].decode("cp932", "replace")
            q = p + 2 + hsize + 0
            skip = 0
            while True:
                nxt = struct.unpack_from("<H", data, q)[0]
                if nxt == 0:
                    q += 2
                    break
                q += nxt
                skip += nxt
            body = q
            csize -= skip
        else:
            nl = data[p + 21]
            name = data[p + 22:p + 22 + nl].decode("cp932", "replace")
            body = p + 2 + hsize
        comp = data[body:body + csize]
        if method == "-lh0-":
            content = comp
        elif method == "-lh5-":
            content = _lh5(comp, osize)
        else:
            raise ValueError("unsupported method " + method)
        files.append((name, content))
        p = body + csize
    return files
