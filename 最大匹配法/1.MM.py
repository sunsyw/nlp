# Maximum Match Method 最大匹配法


class MM:
    def __init__(self):
        self.window_size = 3

    def cut(self, text):
        result = []
        index = 0
        text_length = len(text)
        piece = ''

        dic = ["研究", "研究生", "生命"]
        while text_length > index:
            # range(4,0,-1)
            for size in range(min(self.window_size + index, text_length), index, -1):
                # 4, 3, 2, 1
                piece = text[index: size]
                print('size:', size, piece)
                if piece in dic:
                    index = size - 1
                    break
            index = index + 1  # 第一次结束index = 3
            result.append(piece)
        print(result)
        return result


if __name__ == '__main__':
    mm = MM()
    mm.cut('研究生研究生命')
