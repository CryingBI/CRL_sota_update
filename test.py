# import json

# data = json.load(open("D:\CRL_2\datasets\data_with_marker_tacred.json",'r', encoding='utf-8'))
# print(data)
# for relation in data.keys():
#     print(relation)
e11 = [1, 2]
e22 = [3, 4]
e11 = str(e11)
e22 = str(e22)

with open("entityMarker.txt", 'w', encoding='utf-8') as f:
    f.write(e11 + '\n')
    f.write(e22)

