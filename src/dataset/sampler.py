import time

import numpy as np


class FewshotSampler():
    def __init__(self, data, args, num_episodes=None, state='train'):
        self.data = data
        self.args = args
        self.state = state
        self.num_episodes = num_episodes
        if self.state == 'train':
            self.num_classes = args.train_classes
        elif self.state == 'test':
            self.num_classes = args.test_classes
        elif self.state == 'val':
            self.num_classes = args.val_classes

    def get_sample(self, classes, data):
        examples = {}
        for c in classes:
            examples[c] = []
        support_examples = []
        query_examples = []
        # import pdb
        # pdb.set_trace()
        for d in data:
            if d.label in classes:
                examples[d.label].append(d)
        for c in classes:
            support_examples.extend(examples[c][:self.args.shot])
            query_examples.extend(examples[c][self.args.shot: self.args.shot + self.args.query])
        return support_examples, query_examples

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # 随机抽取类
            sampled_classes = np.random.permutation(self.num_classes)[:self.args.way]
            support, query = self.get_sample(sampled_classes, self.data)

            # import pdb
            # pdb.set_trace()
            yield support, query
