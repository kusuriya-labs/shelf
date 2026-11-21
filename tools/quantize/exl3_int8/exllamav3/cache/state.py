"""Position/slot lifetime for a KV-only model with no recurrent layers."""


class KVState:
    def __init__(self, cache, slot, position, clear=True, test_state=False):
        if cache.recurrent_layers:
            raise ValueError('KVState cannot stand in for recurrent layer state')
        self.cache = cache
        self.slot = slot
        self.position = position
        self.last_history = 0
        self._freed = False

    def post_advance(self):
        pass

    def free(self):
        if not self._freed:
            self.cache.free_list.append(self.slot)
            self._freed = True
