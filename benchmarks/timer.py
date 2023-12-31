import numpy as np
import torch


class Timer:
    def time(
        self, func, dummy_input: torch.Tensor, dummy_label: torch.Tensor, repetitions: int = 300
    ) -> tuple((float, float)):
        # INIT LOGGERS
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        timings = np.zeros((repetitions, 1))

        # GPU-WARM-UP
        for _ in range(10):
            _ = func(dummy_input, dummy_label)

        # MEASURE PERFORMANCE
        for rep in range(repetitions):
            starter.record()
            _ = func(dummy_input, dummy_label)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)

        return mean_syn, std_syn
