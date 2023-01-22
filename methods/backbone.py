import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig

#from dataloaders.sampler import get_tokenizer

class Bert_Encoder(nn.Module):

    def __init__(self, config, out_token=False):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        for param in self.encoder.parameters():
            param.require_grad = False

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size
        self.out_dim = self.output_size

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker', 'maxpooling']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        if self.pattern == 'entity_marker' or 'maxpooling':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        elif self.pattern == 'standard':
            #tokenizer = get_tokenizer(config)
            self.encoder.resize_token_embeddings(config.vocab_size + 1)
            #self.encoder.config.type_vocab_size = 4
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        elif self.pattern == 'entity_marker':
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]): #input size: torch.size([16, 256])
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])

            # input the sample to BERT
            tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
            #size of tokens_output : (16, 256, 768)
            #print("after encode:",self.encoder(inputs))
            output = []
            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                if inputs.device.type in ['cuda']:
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                    instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                else:
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i))
                    instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]))
                output.append(instance_output) # [B,N] --> [B,2,H]
            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
            
            output = self.linear_transform(output)
            #output.size (16, 768) or (1,768)
        elif self.pattern == 'maxpooling':
            e11 = []
            e21 = []
            e12 = []
            e22 = []

            for i in range(inputs.size()[0]): #input size: torch.size([16, 256])
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e12.append(np.argwhere(tokens == 30523)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
                e22.append(np.argwhere(tokens == 30525)[0][0])

            # input the sample to BERT
            tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]

            out = None
            output = []
            for i in range(len(e11)):
                if inputs.device.type in ['cuda']:
                    leng_entity_1 = e12[i] - e11[i]
                    leng_entity_2 = e22[i] - e21[i]
                    # print("leng1", leng_entity_1)
                    # print("leng2", leng_entity_2)
                
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                    instance_output_1 = torch.index_select(instance_output, 1, torch.tensor([e11[i]+x for x in range(1, leng_entity_1)]).cuda())
                    instance_output_2 = torch.index_select(instance_output, 1, torch.tensor([e21[i]+x for x in range(1, leng_entity_2)]).cuda())
                    
                    instance_output_1 = torch.max(instance_output_1, 1).values #(1, 768)
                    #print("is1:",instance_output_1.size())
                    
                    instance_output_2 = torch.max(instance_output_2, 1).values #(1, 768)
                    #print("is2:",instance_output_2.size())

                    out = torch.cat((instance_output_1, instance_output_2), dim=1) #(1, 768*2)
                    #print("out . size:",out.size())
                else:
                    leng_entity_1 = e12[i] - e11[i]
                    leng_entity_2 = e22[i] - e21[i]
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i))
                    instance_output_1 = torch.index_select(instance_output, 1, torch.tensor([e11[i]+x for x in range(1, leng_entity_1)]))
                    instance_output_2 = torch.index_select(instance_output, 1, torch.tensor([e21[i]+x for x in range(1, leng_entity_2)]))
                    instance_output_1 = torch.squeeze(instance_output_1)
                    instance_output_2 = torch.squeeze(instance_output_2)

                    instance_output_1 = torch.max(instance_output_1, 1).values
                    #print("is1:",instance_output_1.size())
                    
                    instance_output_2 = torch.max(instance_output_2, 1).values
                    #print("is2:",instance_output_2.size())

                    out = torch.cat((instance_output_1, instance_output_2), dim=1)
                    
                output.append(out)


            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
            #print("outputs size", output.size())
            output = self.linear_transform(output)
        return output