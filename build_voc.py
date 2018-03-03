from konlpy.tag import Twitter
import tensorflow as tf
import numpy as np
import config
from  Conversation import Conversation

#twitter=Twitter() twitter 사용시.
#def Make_voc(data_path,voc_path):
#    words=[]
#    with open(data_path, 'r', encoding='utf-8') as content_file:
#        for con in content_file: 
#            content = content_file.read()
#            words.extend(twitter.morphs(content))
#        words = list(set(words))
#
#    with open(voc_path, 'w') as vocab_file:
#        for w in words:
#            vocab_file.write(w + '\n')
#
# --twitter 사용시 이 함수 주석을 전부 풀어주고 아래의 함수를 주석처리.

def Make_voc(conv,data_path,voc_path):
    words=[]

    with open(data_path,'r') as f:
        lines=f.read()
        words=conv.Tokenizer(lines)
        words=list(set(words))

    with open(voc_path,'w') as wf:
         for w in words:
            wf.write(w+"\n")
    print("voc complete")
    
def main():
    conv=Conversation()
    Make_voc(conv,config.DATA_PATH,config.VOC_PATH)
#    Make_voc(config.DATA_PATH, config.VOC_PATH) <-twitter 사용시 위를 주석처리하고 이 라인의 주석을 풀어줌.


if __name__=="__main__":
    main()
