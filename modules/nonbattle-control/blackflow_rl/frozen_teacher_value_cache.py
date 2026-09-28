"""Opt-in memoization of unchanged node utilities on frozen public objects."""
from collections import OrderedDict


class FrozenTeacherValueCache:
    def __init__(self,teacher,capacity=32):
        if capacity<1: raise ValueError('Cache capacity must be positive')
        self.teacher=teacher; self.original=teacher._node_value; self.capacity=capacity
        self.entries=OrderedDict(); self.context=None; self.hits=self.misses=0
        teacher._node_value=self.node_value

    def node_value(self,state,node):
        t=self.teacher; sim=t.simulator
        context=(t.config,t.policy_constraints,sim.ruleset,sim.economy.config,sim.economy.catalog,sim.map_generator.config)
        if self.context is None or any(a is not b for a,b in zip(context,self.context)):
            self.entries.clear(); self.context=context
        key=id(state); entry=self.entries.get(key)
        if entry is None:
            entry=(state,{})
            self.entries[key]=entry
            if len(self.entries)>self.capacity: self.entries.popitem(last=False)
        else:
            if entry[0] is not state: raise AssertionError('Strong-reference state cache identity mismatch')
            self.entries.move_to_end(key)
        values=entry[1]; cached=values.get(id(node))
        if cached is not None:
            if cached[0] is not node: raise AssertionError('Strong-reference node cache identity mismatch')
            self.hits+=1; return cached[1]
        self.misses+=1; result=self.original(state,node)
        values[id(node)]=(node,result); return result
