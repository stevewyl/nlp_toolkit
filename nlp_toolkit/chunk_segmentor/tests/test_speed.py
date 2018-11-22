import sys
sys.path.append('../../..')
from nlp_toolkit.chunk_segmentor import Chunk_Segmentor

s = '这是一个能够输出名词短语的分词器，欢迎试用！'


def test_fast():
    cutter = Chunk_Segmentor(mode='fast')
    return list(cutter.cut(s))


def test_accurate():
    cutter = Chunk_Segmentor(mode='accurate')
    return list(cutter.cut(s))


if __name__ == "__main__":
    import cProfile
    cProfile.run("test_accurate()", filename='chunk_speed_accurate.out')
    cProfile.run("test_fast()", filename='chunk_speed_fast.out')
