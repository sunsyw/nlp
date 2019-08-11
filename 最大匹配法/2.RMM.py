# RMM:Reverse Maxmium Match method 逆向最大匹配


class RMM:
    def __init__(self):
        self.window_size = 3

    def cut(self, text):
        result = []
        index = len(text)
        dic = ["研究", "研究生", "生命"]
        piece = ''
        while index > 0:
            for size in range(index - self.window_size, index):
                # 4, 5, 6   # 2, 3, 4
                piece = text[size: index]
                print('size:', size, piece)
                if piece in dic:
                    # index = size + 1
                    index = size
                    break
            # index = index - 1
            result.append(piece)
        result.reverse()
        print(result)
        return result


if __name__ == '__main__':
    rmm = RMM()
    rmm.cut('研究生研究生命')
