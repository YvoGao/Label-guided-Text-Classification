import copy
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class EMBED_BERT(nn.Module):
    def __init__(self, args):
        super(EMBED_BERT, self).__init__()

        self.args = args

        print("{}, Loading pretrained bert".format(
            datetime.datetime.now()), flush=True)

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
        self.model = BertModel.from_pretrained(self.args.pretrained_bert,
                                               cache_dir=self.args.bert_cache_dir)
        self.modelfix = BertModel.from_pretrained(self.args.pretrained_bert,
                                                  cache_dir=self.args.bert_cache_dir)
        # for param in self.modelfix.parameters():
        #     param.requires_grad = False
        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size
        self.template = args.template
        args.embedding_dim = self.embedding_dim

    def forward(self, input_example, query=False):

        
        sentence = [self.args.template.replace(
            "[sentence]", x.text_a)for x in input_example]
        if self.args.dataset == '20newsgroup2' or self.args.dataset == '20newsgroup' or self.args.dataset == 'reuters':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=256, truncation=True)
        elif self.args.dataset == 'reuters':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=64, truncation=True)
        elif self.args.dataset == 'amazon2' or self.args.dataset == 'amazon':
            inputs = self.tokenizer.batch_encode_plus(sentence, return_tensors='pt',padding='max_length', max_length=128, truncation=True)
        else:
            inputs = self.tokenizer.batch_encode_plus(
                sentence, return_tensors='pt', padding=True)
        inputs.to(self.args.device)
        outputs = self.model(**inputs)
        labels = [x.text_b for x in input_example]
        label_inputs = self.tokenizer.batch_encode_plus(
            labels, return_tensors='pt', padding=True).to(self.args.device)
        label_outputs = self.model(
            **label_inputs).last_hidden_state.mean(dim=1)
        # label_outputs = self.modelfix(
        #     **label_inputs).last_hidden_state.mean(dim=1)
        # label_outputs = self.modelfix(**label_inputs).last_hidden_state[:,0,:]

        predictions = torch.zeros(
            [inputs['input_ids'].shape[0], self.embedding_dim]).to(self.args.device)
        mask_token_index = torch.where(
            inputs['input_ids'] == self.tokenizer.mask_token_id)[1]

        if self.args.pool == 'prompt':
            # prompt
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i,
                                                           mask_token_index[i], :]
        elif self.args.pool == 'cls':
            # cls
            for i in range(len(predictions)):
                predictions[i] = outputs.last_hidden_state[i, 0, :]
        elif self.args.pool == 'avg':
            # avgpool
            predictions = outputs.last_hidden_state.mean(dim=1)
        return predictions, label_outputs
