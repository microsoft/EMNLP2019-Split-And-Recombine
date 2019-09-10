# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BindTrieNode(object):
    def __init__(self):
        self.children = {}
        self.value = None


class BindTrie:
    def __init__(self):
        self.root = BindTrieNode()

    def add(self, words, match_type):
        node = self.root
        n = len(words)
        for i in range(n):
            if words[i] not in node.children:
                new_node = BindTrieNode()
                node.children[words[i]] = new_node
                node = new_node
            else:
                node = node.children[words[i]]
            if i == n - 1:
                if node.value is None:
                    node.value = []
                node.value.append(match_type)

    def find(self, words):
        node = self.root
        for word in words:
            if word not in node.children:
                return False, None
            node = node.children[word]
        if node.value is not None:
            return True, node.value
        else:
            return True, None
