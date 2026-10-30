"""Action-aligned card identities for the fixed observe2/action ABI."""


def build_identity_head(nn, torch, cfg):
    class ActionIdentityHead(nn.Module):
        def __init__(self):
            super().__init__()
            # Public ABI: board[19:26], hand[26:36], shop[36:44],
            # pending choices[44:60]. Seven queries share contextual features.
            self.query = nn.Linear(cfg.hidden + 2 * cfg.card_emb, 7 * cfg.card_emb)
            nn.init.zeros_(self.query.weight)
            nn.init.zeros_(self.query.bias)

        def forward(self, hidden, embeddings, cards, pooled):
            board = embeddings[:, 19:26]
            hand = embeddings[:, 26:36]
            shop = embeddings[:, 36:44]
            choices = embeddings[:, 44:60]
            owned_mask = (cards[:, 19:36] != 0).float().unsqueeze(-1)
            owned = (embeddings[:, 19:36] * owned_mask).sum(1)
            owned = owned / owned_mask.sum(1).clamp_min(1.0)
            queries = self.query(torch.cat([hidden, owned, pooled], -1))
            queries = queries.reshape(-1, 7, cfg.card_emb)

            def score(items, group):
                return (items * queries[:, group:group + 1]).sum(-1) / cfg.card_emb ** 0.5

            zeros = hidden.new_zeros((hidden.shape[0], 245))
            buy = score(shop, 0)
            sell = score(board, 1)
            activate = score(board, 2)
            choose = score(choices, 3)
            play = score(hand, 4).unsqueeze(-1).expand(-1, -1, 8).reshape(-1, 80)
            magnetize = (score(hand, 5).unsqueeze(-1) + score(board, 6).unsqueeze(1))
            return torch.cat([zeros[:, :8], buy, sell, activate, choose, play,
                              magnetize.reshape(-1, 70), zeros[:, 196:]], -1)

    return ActionIdentityHead()
