"""Bounded memoization for immutable public policy states.

This mixin has no controller base. The route controller opts in explicitly;
other controllers retain their existing evaluation path. Keeping the original
objects in each slot makes identity comparison safe from object-id reuse.
"""


class PublicPolicyCacheMixin:
    observation_cache_capacity = 8

    def _check_policy_cache_context(self):
        context = (self.config, self.policy_constraints, self.simulator.economy.config,
                   self.simulator.map_generator.config)
        previous = getattr(self, '_policy_cache_context', None)
        if previous is None or any(old is not new for old, new in zip(previous, context)):
            self._policy_cache_context = context
            self._observation_pairs = []
            self._value_caches = {}

    def observable_state(self, state):
        self._check_policy_cache_context()
        cache = self._observation_pairs
        for index, (original, projected) in enumerate(cache):
            if original is state:
                cache.append(cache.pop(index))
                return projected
        # A projected state is a distinct input and must itself be projected
        # normally. No assertion that belief(belief(s)) == belief(s) is made.
        projected = self.simulator.belief_state(state)
        cache.append((state, projected))
        if len(cache) > self.observation_cache_capacity:
            del cache[0]
        return projected

    def _memo_value(self, name, state, key, calculate):
        self._check_policy_cache_context()
        cache = self._value_caches.setdefault(name, [])
        values = None
        for index, (original, entries) in enumerate(cache):
            if original is state:
                cache.append(cache.pop(index))
                values = entries
                break
        if values is None:
            values = {}
            cache.append((state, values))
            if len(cache) > self.observation_cache_capacity:
                del cache[0]
        if key not in values:
            values[key] = calculate()
        return values[key]

    def _exchange_value(self, state, instance):
        calculate = super()._exchange_value
        return self._memo_value('exchange', state, instance, lambda: calculate(state, instance))

    def _held_part_value(self, state, instance):
        calculate = super()._held_part_value
        return self._memo_value('held_part', state, instance, lambda: calculate(state, instance))

    def _concept_opportunities(self, state, target_type, *, include_future=True):
        calculate = super()._concept_opportunities
        return self._memo_value('concept_opportunities', state, (target_type, include_future),
            lambda: calculate(state, target_type, include_future=include_future))
