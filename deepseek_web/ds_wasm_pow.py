import wasmtime
import numpy as np


class DS_WasmPow:
    def __init__(self, wasm_file_path):
        # Initialize the WebAssembly engine, store, and load the module
        self.engine = wasmtime.Engine()
        self.store = wasmtime.Store(self.engine)

        with open(wasm_file_path, "rb") as f:
            wasm_bytes = f.read()
        self.module = wasmtime.Module(self.engine, wasm_bytes)
        self.linker = wasmtime.Linker(self.engine)
        self.instance = self.linker.instantiate(self.store, self.module)

        # Exported functions and memory
        self.memory = self.instance.exports(self.store)["memory"]
        self.alloc_func = self.instance.exports(self.store)["__wbindgen_export_0"]
        self.wasm_solve = self.instance.exports(self.store)["wasm_solve"]
        self.stack_pointer_func = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"]

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        # Helper function to write data to WebAssembly memory
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.alloc_func(self.store, length, 1)

        memory_view = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            memory_view[ptr + i] = byte

        return ptr, length

    def calculate_answer(self, challenge, salt, difficulty, expire_at):
        prefix = f"{salt}_{expire_at}_"
        retptr = self.stack_pointer_func(self.store, -16)

        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)

            self.wasm_solve(
                self.store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty),
            )

            memory_view = self.memory.data_ptr(self.store)
            status = int.from_bytes(bytes(memory_view[retptr : retptr + 4]), byteorder="little", signed=True)
            if status == 0:
                return None

            value_bytes = bytes(memory_view[retptr + 8 : retptr + 16])
            value = np.frombuffer(value_bytes, dtype=np.float64)[0]

            return int(value)
        finally:
            self.stack_pointer_func(self.store, 16)


if __name__ == "__main__":
    wasm_file = "sha3_wasm_bg.7b9ca65ddd.wasm"
    ds_wasm_pow = DS_WasmPow(wasm_file)
    challenge = "2ee17d427355d5d0bb1056a14d8ed6982f117db6c2e4046bc05f53c1546876b6"
    salt = "e071bdd62e1dcb455990"
    difficulty = 144000
    expire_at = 1736928349211

    answer = ds_wasm_pow.calculate_answer(challenge, salt, difficulty, expire_at)
    print("Answer:", answer)  # 38385
