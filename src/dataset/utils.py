import torch
import datetime
from typing import Union
from typing import *
def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now(), s),
          flush=True)


def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, max_len=None):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''

    for k in keys:
        new_data[k] = old_data[k][idx]
        if max_len is not None and len(new_data[k].shape) > 1:
            new_data[k] = new_data[k][:,:max_len]

    return new_data


# class InputExample(object):
#     """A single training/test example for simple sequence classification."""

#     def __init__(self, text_token_id, text_raw, label_id, label_token_id, label_raw,
#                  domain_id, domain_raw=None, head_pos=None, tail_pos=None):
#         """Constructs a InputExample.
#         Args:
#             text_token_id: int. The BERT token id of the text
#             text_raw: string. The untokenized text of the first sequence. For single
#             sequence tasks, only this sequence must be specified.
#             label_id: int. The label id for the sample.
#             label_token_id: int. The BERT token id of the label
#             label_raw: string. The label of the example. This should be
#             specified for train and dev examples, but not for test examples.
#             domain_id: (Optional) int. The domain id for the sample, used for the multi-domain dataset.
#             domain_raw: (Optional) string. The domain name for the sample.
#             head_pos: (Optional) int. Used for FewRel dataset.
#             tail_pos: (Optional) int. Used for FewRel dataset.
#         """

#         self.text_token_id = text_token_id
#         self.text_raw = text_raw
#         self.label_id = label_id
#         self.label_token_id = label_token_id
#         self.label_raw = label_raw
#         self.domain_id = domain_id
#         self.domain_raw = domain_raw
#         self.head_pos = head_pos
#         self.tail_pos = tail_pos
class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid = None,
                 text_a = "",
                 text_b = "",
                 label = None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str, List[str]]] = None
                ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text