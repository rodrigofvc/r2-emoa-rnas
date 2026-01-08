import numpy as np


class OFASearchSpace:
    def __init__(self):
        self.num_blocks = 5  # number of blocks
        self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
        self.exp_ratio = [3, 4, 6]  # expansion rate
        #self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
        self.depth = [1]
        self.resolution = list(range(192, 257, 4))  # input image resolutions

    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, r=None):
        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.kernel_size if ks is None else ks
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        r = self.resolution if r is None else r

        data = []
        for n in range(n_samples):
            # first sample layers
            depth = np.random.choice(d, nb, replace=True).tolist()
            # then sample kernel size, expansion rate and resolution
            kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
            exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()
            resolution = int(np.random.choice(r))

            data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution})
        return data

    def initialize(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = [
            self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                        d=[min(self.depth)], r=[min(self.resolution)])[0],
            self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                        d=[max(self.depth)], r=[max(self.resolution)])[0]
        ]
        data.extend(self.sample(n_samples=n_doe - 2))
        return data

    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                if counter < len(x):
                    new_x.append(x[counter])
                    counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        if isinstance(config, np.ndarray):
            config = config[0]
        # encode config ({'ks': , 'd': , etc}) to integer bit-string [1, 0, 2, 1, ...]
        x = []
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        kernel_size = [np.argwhere(_x == np.array(self.kernel_size))[0, 0] for _x in config['ks']]
        exp_ratio = [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['e']]

        kernel_size = self.pad_zero(kernel_size, config['d'])
        exp_ratio = self.pad_zero(exp_ratio, config['d'])

        for i in range(len(depth)):
            x = x + [depth[i]] + kernel_size[i * max(self.depth):i * max(self.depth) + max(self.depth)] \
                + exp_ratio[i * max(self.depth):i * max(self.depth) + max(self.depth)]
        x.append(np.argwhere(config['r'] == np.array(self.resolution))[0, 0])

        return x

    def decode(self, x):
        """
        remove un-expressed part of the chromosome
        assumes x = [block1, block2, ..., block5, resolution, width_mult];
        block_i = [depth, kernel_size, exp_rate]
        """
        depth, kernel_size, exp_rate = [], [], []
        print('>>>>>>>>>>>   Decoding architecture: ', x, type(x), x.shape)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        x = x.astype(int)

        for i in range(0, len(x) - 2, 9):
            if i < len(x):
                x = np.squeeze(x)
                depth_index = x[i]
                if depth_index < len(self.depth):
                    depth.append(self.depth[depth_index])
                    if i + 1 < len(x):
                        ks_start = i + 1
                        ks_end = ks_start + self.depth[depth_index]
                        if ks_end < len(x) and ks_start < len(x):
                            if len(self.kernel_size) > max(x[ks_start:ks_end]):
                                kernel_size.extend(np.array(self.kernel_size)[x[ks_start:ks_end]].tolist())
                    if i + 5 < len(x):
                        exp_start = i + 5
                        exp_end = exp_start + self.depth[depth_index]
                        if exp_end < len(x) and exp_start < len(x):
                            if len(self.exp_ratio) > max(x[exp_start:exp_end]):
                                exp_rate.extend(np.array(self.exp_ratio)[x[exp_start:exp_end]].tolist())
            """
            if i < len(x):
                depth.append(self.depth[x[i]])
            if i + 1 + self.depth[x[i]] < len(x) and i + 1 < len(x):
                kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
            if i + 5 + self.depth[x[i]] < len(x) and i + 5 < len(x):
                exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())
            """
        return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]]}

