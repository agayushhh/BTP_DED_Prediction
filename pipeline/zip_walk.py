"""Walk ZIP local headers front-to-back and extract entries (Deflate64 via inflate64).

Works on a partially downloaded zip (central directory not needed): stops at the first entry whose
data is not yet fully present. Usage: python -I zip_walk.py <zip> <out_dir> [<contiguous_bytes>]
"""
import struct, zlib, os, sys
import inflate64


def walk(path, limit=None):
    limit = limit or os.path.getsize(path)
    with open(path, 'rb') as fh:
        pos = 0
        while pos + 30 < limit:
            fh.seek(pos)
            h = fh.read(30)
            if h[:4] != b'PK\x03\x04':
                return  # central directory reached
            _, _, flag, meth, _, _, _, cs, us, nl, el = struct.unpack('<IHHHHHIIIHH', h)
            name = fh.read(nl).decode('utf8', 'replace')
            extra = fh.read(el)
            if cs == 0xFFFFFFFF:  # zip64 sizes live in the extra field
                i = 0
                while i < len(extra):
                    hid, sz = struct.unpack('<HH', extra[i:i + 4])
                    if hid == 1:
                        us, cs = struct.unpack('<QQ', extra[i + 4:i + 20])
                    i += 4 + sz
            assert not (flag & 8 and cs == 0), 'streamed entry; need central directory'
            start = pos + 30 + nl + el
            if start + cs > limit:
                return
            raw = fh.read(cs)
            if meth == 9:
                data = inflate64.Inflater().inflate(raw)
            elif meth == 8:
                data = zlib.decompress(raw, -15)
            else:
                data = raw
            yield name, data
            pos = start + cs


def contiguous_bytes(path, chunk=8 * 2**20):
    """Bytes available from the start, given download.py's <file>.done chunk log."""
    if not os.path.exists(path + '.done'):
        return os.path.getsize(path)
    done = set(int(x) for x in open(path + '.done'))
    k = 0
    while k in done:
        k += 1
    return k * chunk


if __name__ == '__main__':
    z, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, data in walk(z, contiguous_bytes(z)):
        if name.endswith('/'):
            continue
        p = os.path.join(out, os.path.basename(name))
        if not os.path.exists(p):
            open(p, 'wb').write(data)
        n += 1
    print(n, 'files')
