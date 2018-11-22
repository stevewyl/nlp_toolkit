"""Trie树结构"""


class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.is_word = False


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中

    def starts_with(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
        Returns words started with prefix
        :param prefix:
        :return: words (list)
        """
        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.extend(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list

        words = []
        if not self.starts_with(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)


if __name__ == '__main__':
    tree = Trie()
    tree.insert('深度学习')
    tree.insert('深度神经网络')
    tree.insert('深度网络')
    tree.insert('机器学习')
    tree.insert('机器学习模型')
    print(tree.search('深度学习'))
    print(tree.search('机器学习模型'))
    print(tree.get_start('深度'))
    print(tree.get_start('深度网'))
