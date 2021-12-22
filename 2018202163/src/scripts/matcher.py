import logging
from modules import * 


class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_input=0.3, finetune=False,
                 dropout_neighbors=0.0,
                 device=torch.device("cpu")):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.NeighborAggregator = AttentionSelectContext(dim=embed_dim, dropout=dropout_neighbors)

    def neighbor_encoder_mean(self, connections, num_neighbors):
        """
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()

    def neighbor_encoder_soft_select(self, connections_left, connections_right, head_left, head_right):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        relations_left = connections_left[:, :, 0].squeeze(-1)
        entities_left = connections_left[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).squeeze(-1)  # [b, max]

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (batch, 200, embed_dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (batch, 200, embed_dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).squeeze(-1)  # [b, max]

        left = [head_left, rel_embeds_left, ent_embeds_left]
        right = [head_right, rel_embeds_right, ent_embeds_right]
        output = self.NeighborAggregator(left, right, mask_matrix_left, mask_matrix_right)
        return output

    def forward(self, entity, entity_meta=None):
        '''
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         '''
        if entity_meta is not None:
            entity = self.symbol_emb(entity)
            entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
            entity_left, entity_right = self.neighbor_encoder_soft_select(entity_left_connections,
                                                                          entity_right_connections,
                                                                          entity_left, entity_right)
        else:
            # no_meta
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
        return entity_left, entity_right

class RelationRepresentation(nn.Module):
    def __init__(self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3,
                                                  with_pos=True)

    def forward(self, left, right):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.RelationEncoder(left, right)
        return relation

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

#encoder for support pairs
class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)

#LSTM networks for multi-step matching
class QueryEncoder(nn.Module):
    """docstring for QueryEncoder"""
    def __init__(self, input_dim, process_step=4):
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        # self.batch_size = batch_size
        self.process = nn.LSTMCell(input_dim, 2*input_dim)


    def forward(self, support, query):
        '''
        support: (few, support_dim)
        query: (batch_size, query_dim)
        support_dim = query_dim

        return:
        (batch_size, query_dim)
        '''
        assert support.size()[1] == query.size()[1]

        if self.process_step == 0:
            return query

        batch_size = query.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()
        #multi-step matching
        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:,:self.input_dim] # (batch_size, query_dim)
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support) # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        # return h_r_[:, :self.input_dim]
        return h

class Matcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_layers=0.1, dropout_input=0.3,
                 dropout_neighbors=0.0,
                 finetune=False, num_transformer_layers=6, num_transformer_heads=4,
                 device=torch.device("cpu")
                 ):
        super(Matcher, self).__init__()
        self.EntityEncoder = EntityEncoder(embed_dim, num_symbols,
                                           use_pretrain=use_pretrain,
                                           embed=embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors,
                                           finetune=finetune, device=device)
        self.RelationRepresentation = RelationRepresentation(emb_dim=embed_dim,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.Prototype = SoftSelectPrototype(embed_dim * num_transformer_heads)

        self.embed_dim = embed_dim
        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.dropout = nn.Dropout(0.5)
        d_model = embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, 0.2)
        self.query_encoder = QueryEncoder(d_model, 4)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    #for each support pair,get matching score with query pair
    def getmatchscore(self,support,query,support_meta,query_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections_list, support_left_degrees_list, support_right_connections_list, support_right_degrees_list = support_meta
        
        
        matching_scores = 0
        for support_left_connections, support_left_degrees, support_right_connections, support_right_degrees \
        in zip(support_left_connections_list, support_left_degrees_list, support_right_connections_list, support_right_degrees_list):
            query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
            query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

            support_left = self.neighbor_encoder(support_left_connections.unsqueeze(0), support_left_degrees.unsqueeze(0))
            support_right = self.neighbor_encoder(support_right_connections.unsqueeze(0), support_right_degrees.unsqueeze(0))

            query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
            support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh
            
            #encode support pair and query pair
            support_g = self.support_encoder(support_neighbor)
            query_g = self.support_encoder(query_neighbor)

            support_g =torch.mean(support_g,dim = 0,keepdim=True)
            query_f = self.query_encoder(support_g,query_g)
            matching_scores = torch.matmul(query_f, support_g.t()).squeeze()
        return matching_scores

    def forward(self, support, query, false=None, isEval=False, support_meta=None, query_meta=None, false_meta=None,neighbor_info=None,neighbor_info_metas=None):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """
        if not isEval:
            
            support_neighbor,query_neighbor = neighbor_info
            support_neighbor_metas,query_neighbor_metas = neighbor_info_metas
            
            self.EntityEncoder(support_neighbor,support_neighbor_metas)
            self.EntityEncoder(query_neighbor,query_neighbor_metas)

            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)
            false_r = self.EntityEncoder(false, false_meta)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            false_r = self.RelationRepresentation(false_r[0], false_r[1])

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)

            #scores for positive samples an negative samples respectively
            matching_scores_p = self.getmatchscore(support, query, support_meta, query_meta)
            matching_scores_f = self.getmatchscore(support, false, support_meta, false_meta)
            # Mixed Macthing Scores
            # weighted sum 
            positive_score += matching_scores_p * 0.2
            negative_score += matching_scores_f * 0.2
        else:
            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            
            center_q = self.Prototype(support_r, query_r)

            positive_score = torch.sum(query_r * center_q, dim=1)
            matching_scores = self.getmatchscore(support, query, support_meta, query_meta)
            negative_score = None

            # Mixed Macthing Scores
            positive_score += matching_scores * 0.2
        return positive_score, negative_score
