class Normalizer:
    def __init__(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean
    
    def get_mean(self):
        return self.mean
    
    def get_std(self):
        return self.std
    
    def __str__(self):
        return f"Normalizer(mean={self.mean}, std={self.std})"