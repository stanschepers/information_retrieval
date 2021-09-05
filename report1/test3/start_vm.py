import lucene

try:
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
except Exception as e:
    print(e)
