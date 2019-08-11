import os
import codecs
import csv


corpus_name = "data"
corpus = os.path.join("data", corpus_name)


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# printLines(os.path.join(corpus, 'movie_lines.txt'))


def loadLines(fileName, fields):
    """
    :param fileName:
    :param fields:
    :return: lineObj = {lineID:L1045, characterID:u0, movieID:m0, character:BIANCA, text:They do not!}
             lines = {lineID: lineObj
                      lineID: lineObj}
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


def loadConversations(fileName, lines, fields):
    """
    :param fileName:
    :param lines:
    :param fields:
    :return: convObj = {character1ID:u0, character2ID:u2, movieID:m0, utteranceIDs:['L194', 'L195', 'L196', 'L197'], lines:lineObj}
             conversations = [convObj, convObj, ...]
    """

    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj['utteranceIDs'])
            convObj['lines'] = []
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines']) - 1):
            inputline = conversation['lines'][i]['text'].strip()
            targetline = conversation['lines'][i+1]['text'].strip()
            if inputline and targetline:
                qa_pairs.append([inputline, targetline])
    return qa_pairs


if __name__ == '__main__':
    datafile = os.path.join(corpus, 'formatted_movie_lines.txt')
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

    # lines = {}
    # conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)
