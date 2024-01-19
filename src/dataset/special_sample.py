import time
import numpy as np
import torch
import torch.nn.functional as F
from queue import Queue
import dataset.utils as utils
from dataset.loader import get_label_dict


def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def pos_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def cos_sim(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    normalized_instances = F.normalize(instances, dim=1)
    normalized_class_proto = F.normalize(class_proto, dim=1)

    cosine_sim = torch.mm(normalized_instances, normalized_class_proto.t())
    return cosine_sim


def dot_similarity(XS, XQ):
    return torch.matmul(XS, XQ.t())


def reidx_y(args, YS, YQ):
    '''
        Map the labels into 0,..., way
        @param YS: batch_size
        @param YQ: batch_size
        @return YS_new: batch_size
        @return YQ_new: batch_size
    '''
    unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
    unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

    if len(unique1) != len(unique2):
        raise ValueError(
            'Support set classes are different from the query set')

    if len(unique1) != args.way:
        print("unique1", unique1)
        print("inv_S", inv_S)
        raise ValueError(
            'Support set classes are different from the number of ways')

    if int(torch.sum(unique1 - unique2).item()) != 0:
        raise ValueError(
            'Support set classes are different from the query set classes')

    Y_new = torch.arange(start=0, end=args.way, dtype=unique1.dtype,
                         device=unique1.device)

    return Y_new[inv_S], Y_new[inv_Q]


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


# 得到每一个类，和类内对应的相似矩阵
# 根据类的名字判断相似性
def pre_calculate(train_data, net, args, state):
    with torch.no_grad():
        if state == 'train':
            all_classes = args.train_classes
        elif state == 'test':
            all_classes = args.test_classes
        elif state == 'val':
            all_classes = args.val_classes
        label_dict = get_label_dict(args)
        id2labelname = {v: k for k, v in label_dict.items()}
        labels = [id2labelname[x] for x in all_classes]
        label_inputs = net.tokenizer.batch_encode_plus(
            labels, return_tensors='pt', padding=True).to(args.device)
        label_ebd = net.model(**label_inputs).last_hidden_state.mean(dim=1)

        # metrix of labelname similarty

        dist_metrix = dot_similarity(label_ebd, label_ebd)

        for i, d in enumerate(dist_metrix):
            if i == 0:
                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat(
                    (dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        prob_metrix = F.softmax(dist_metrix_nodiag, dim=1)  # [10, 9]
        prob_metrix = prob_metrix.cpu().numpy()

        # 生成sample样本时候的概率矩阵
        example_prob_metrix = []
        for i, label in enumerate(all_classes):
            examples = []
            for x in train_data:
                if x.label == label:
                    examples.append(x)
            sentence = [net.args.template.replace(
                "[sentence]", x.text_a)for x in examples]
            sentence_ebds = []
            # for s in sentence:
            for li in range(0, len(sentence), 150):
                s = sentence[li: li + 150]
                inputs = net.tokenizer.batch_encode_plus(s, return_tensors='pt', padding=True).to(args.device)
                outputs = net.model(**inputs)
                tmp_ebd = torch.zeros(
                    [inputs['input_ids'].shape[0], net.embedding_dim]).to(net.args.device)
                mask_token_index = torch.where(
                    inputs['input_ids'] == net.tokenizer.mask_token_id)[1]
                for j in range(len(tmp_ebd)):
                    tmp_ebd[j] = outputs.last_hidden_state[j,
                                                           mask_token_index[j], :]
                sentence_ebds.append(tmp_ebd)

            # import pdb
            # pdb.set_trace()
            # 将sentence_ebds转成tensor（n, 768）
            sentence_ebd = torch.cat(sentence_ebds, 0)
            example_prob_metrix_one = dot_similarity(label_ebd[i].view(
                (1, -1)), sentence_ebd)
            example_prob_metrix_one = F.softmax(
                example_prob_metrix_one, dim=1)  # [1, 1000]

            example_prob_metrix_one = example_prob_metrix_one.cpu().numpy()

            example_prob_metrix.append(example_prob_metrix_one)

        return prob_metrix, example_prob_metrix


class SerialSampler():

    def __init__(self, data, args, num_episodes=None, state='train', example_prob_metrix=None):
        self.data = data
        self.args = args
        self.state = state
        self.num_episodes = num_episodes
        self.example_prob_metrix = example_prob_metrix
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
        for d in data:
            if d.label in classes:
                examples[d.label].append(d)
        if self.example_prob_metrix is None:
            for c in classes:
                support_examples.extend(examples[c][:self.args.shot])
                query_examples.extend(
                    examples[c][self.args.shot: self.args.shot + self.args.query])
        else:
            for c in classes:
                # 得到了概率高的id
                tmp = np.random.choice(len(
                    examples[c]), self.args.shot + self.args.query, p=self.example_prob_metrix[c][0], replace=False)
                # 选择概率高的几个例子
                for i in range(self.args.shot):
                    support_examples.append(examples[c][tmp[i]])
                for i in range(self.args.shot, self.args.query):
                    query_examples.append(examples[c][tmp[i]])

        return support_examples, query_examples

    def get_epoch(self, sampled_classes):
        for _ in range(self.num_episodes):
            # 随机抽取类
            # sampled_classes = np.random.permutation(
            #     self.num_classes)[:self.args.way]
            support, query = self.get_sample(sampled_classes, self.data)

            yield support, query


def task_sampler(data, args, classes_sample_p=None):
    all_classes = args.train_classes
    num_classes = len(all_classes)
    _, id_metrix = np.mgrid[0: num_classes: 1,
                            0: num_classes: 1]  # [N, N]每行都是0~(N-1)
    id_metrix = id_metrix[~np.eye(id_metrix.shape[0], dtype=bool)].reshape(
        id_metrix.shape[0], -1)  # 去掉了对角线元素

    # sample classes
    if classes_sample_p is None:
        temp = np.random.permutation(num_classes)
        sampled_classes = temp[:args.way]
    else:
        class_names_num = []
        class_name_num = np.random.choice(len(all_classes), 1)
        a = class_name_num[0]
        class_names_num.append(a)
        p = classes_sample_p[a]
        for i in range(args.way - 1):
            class_name_num = np.random.choice(
                id_metrix[a], 1, p=p, replace=False)
            a = class_name_num[0]
            if a in class_names_num:
                t1 = np.arange(len(all_classes))
                t2 = []
                for k in t1:
                    if k not in class_names_num:
                        t2.append(k)
                # print("t1", t1)
                a = np.random.choice(t2, 1)[0]
            class_names_num.append(a)
            p = (p + classes_sample_p[a]) / 2

        sampled_classes = class_names_num

    source_classes = None

    return sampled_classes, source_classes
