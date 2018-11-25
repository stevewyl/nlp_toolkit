import sys
# sys.path.append('../../..')
from nlp_toolkit.chunk_segmentor import Chunk_Segmentor

mode = sys.argv[1]

if mode == 'short':
    text = '这是一个能够输出名词短语的分词器，欢迎试用！'
elif mode == 'long':
    text = '主要配合UI设计师100%还原设计图，使用前端技术解决各大浏览器兼容问题，使用HTML5+css3完成页面优化及提高用户体验，www.s.com使用bootstrap、jQuery完成界面效果展示，使用JavaScript完成页面功能展示，并且，在规定的时间内提前完成任务大大提高了工作的效率'


def load_fast():
    return Chunk_Segmentor(mode='fast')


def test_fast():
    return list(CUTTER.cut([text] * 10000))


def load_accurate():
    return Chunk_Segmentor(mode='accurate')


def test_accurate():
    return list(CUTTER.cut([text] * 10000))


if __name__ == "__main__":
    import cProfile
    global CUTTER
    CUTTER = load_accurate()
    cProfile.run("test_accurate()", filename='chunk_speed_accurate_%s.out' % mode)
    CUTTER = load_fast()
    cProfile.run("test_fast()", filename='chunk_speed_fast_%s.out' % mode)
