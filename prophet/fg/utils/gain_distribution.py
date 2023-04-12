class GainDistribution:

    class Bucket:

        def __init__(self, gain, weight):
            self.gain = gain
            self.weight = weight
            self.age = 0

        def __str__(self):
            return '{}:{}:{}'.format(self.gain, self.weight, self.age)

        def __repr__(self):
            return self.__str__()

    def __init__(self, max_age, decay, min_weight):
        self.max_age = max_age
        self.decay = decay
        self.min_weight = min_weight
        self.buckets = []

    def update(self, gain, weight):
        new_buckets = []
        for bucket in self.buckets:
            bucket.gain += gain
            bucket.weight *= self.decay
            bucket.age += 1
            if bucket.weight >= self.min_weight and bucket.age < self.max_age:
                new_buckets.append(bucket)

        new_bucket = GainDistribution.Bucket(0, weight)
        new_buckets.append(new_bucket)
        self.buckets = new_buckets

    def cdf(self):
        buckets = sorted(self.buckets, key=lambda x: x.gain)
        z_total = 0
        z_left = 0
        for bucket in buckets:
            mass = bucket.weight
            z_total += mass
            if bucket.gain < 0:
                z_left += mass
        return 1 - z_left / z_total

    def rv(self):
        if len(self.buckets) <= 1:
            return 0.5

        z_total = 0
        z_reversed = 0
        for i in range(len(self.buckets)):
            for j in range(i + 1, len(self.buckets)):
                b1 = self.buckets[i]
                b2 = self.buckets[j]
                z_total += 1
                if b1.gain == b2.gain:
                    z_reversed += 0.5
                elif not((b1.age < b2.age) ^ (b1.gain < b2.gain)):
                    z_reversed += 1
        return z_reversed / z_total

