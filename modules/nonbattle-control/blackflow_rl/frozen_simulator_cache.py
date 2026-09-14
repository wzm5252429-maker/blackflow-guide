"""Bounded memoization of pure reads in an explicitly frozen simulator.

No transitions, chance draws or policy outputs are cached. Strong references
prevent object-id reuse. Configuration replacement clears every cached result.
This is an opt-in execution optimization, not a simulator rule change.
"""
from collections import Counter,OrderedDict
from hashlib import sha256
from pathlib import Path


class FrozenSimulatorReadCache:
    def __init__(self,simulator,capacity=128,node_capacity=2048):
        if hasattr(simulator,'_frozen_read_cache'):
            raise ValueError('Read cache already installed')
        self.simulator=simulator; self.capacity=capacity; self.node_capacity=node_capacity
        self.original={name:getattr(simulator,name) for name in ('available_options','legal_actions','belief_state','_expected_unentered_node')}
        self.cache={name:OrderedDict() for name in self.original}; self.calls=Counter(); self.hits=Counter()
        self.context=self._context()
        for name in ('available_options','legal_actions','belief_state'):
            setattr(simulator,name,self._wrapper(name))
        simulator._expected_unentered_node=self.expected_node
        simulator._frozen_read_cache=self

    @property
    def implementation_sha256(self): return sha256(Path(__file__).read_bytes()).hexdigest()

    def _context(self):
        sim=self.simulator
        return (sim.ruleset,sim.map_generator.config,sim.economy.config,sim.simulation_profile)

    def _check(self):
        context=self._context()
        if any(a is not b for a,b in zip(context,self.context)):
            for cache in self.cache.values(): cache.clear()
            self.context=context

    def _wrapper(self,name):
        def wrapped(state):
            self._check(); self.calls[name]+=1; cache=self.cache[name]; key=id(state)
            entry=cache.get(key)
            if entry is not None and entry[0] is state:
                self.hits[name]+=1; cache.move_to_end(key); return entry[1]
            value=self.original[name](state); cache[key]=(state,value)
            if len(cache)>self.capacity: cache.popitem(last=False)
            return value
        return wrapped

    def expected_node(self,node,node_type,event_options):
        self._check(); name='_expected_unentered_node'; self.calls[name]+=1
        cache=self.cache[name]; key=(id(node),node_type,event_options); entry=cache.get(key)
        if entry is not None and entry[0] is node:
            self.hits[name]+=1; cache.move_to_end(key); return entry[1]
        value=self.original[name](node,node_type,event_options); cache[key]=(node,value)
        if len(cache)>self.node_capacity: cache.popitem(last=False)
        return value

    def uninstall(self):
        for name,value in self.original.items(): setattr(self.simulator,name,value)
        del self.simulator._frozen_read_cache

    def statistics(self):
        return {'calls':dict(self.calls),'hits':dict(self.hits),'sizes':{k:len(v) for k,v in self.cache.items()},
                'implementation_sha256':self.implementation_sha256}
