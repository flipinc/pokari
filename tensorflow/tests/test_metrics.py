import math

from metrics.error_rate import cer, wer


class TestMetrics:
    def test_cer(self):
        predicted = [b"the tensorflow lite converter generates a tensorflow lite model"]
        label = [b"da tensaflow aite convata generat a tensorflow aite modell"]

        distance, length = cer(predicted, label)

        assert math.floor((distance / length) * 100) == 24

    def test_wer(self):
        predicted = [b"the tensorflow lite converter generates a tensorflow lite model"]
        label = [b"da tensaflow aite convata generat a tensorflow aite modell"]

        distance, length = wer(predicted, label)

        assert math.floor((distance / length) * 100) == 77
