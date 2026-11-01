import config
import multiprocessing as mp
from contextlib import redirect_stdout
from common import data
from fox_diagnostic.fox_diagnostic import multiply_mp


class Tee:
    """Write stdout to both the terminal and the diagnostic log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    # Load one saved pair: G0 pair index 7 (the 2048 x 2048 pair).
    A, B = data.load_group(0)[-1]
    n = len(A)

    output_path = config.ROOT / "fox_diagnostic" / "fox_diagnostic_output.txt"
    with output_path.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(__import__("sys").stdout, output_file)):
            for p in range(1, 7):
                print(f"\n{'=' * 65}\nFox diagnostic: n={n}, p={p}\n{'=' * 65}")
                with mp.get_context(config.START_METHOD).Pool(p) as pool:
                    C, results = multiply_mp(A, B, pool, p)
