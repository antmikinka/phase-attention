class ByteTokenizer:
    """UTF-8 bytes 0..255. No external vocabulary; loss in nats/byte -> bits-per-byte = loss/ln 2."""
    vocab_size = 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
