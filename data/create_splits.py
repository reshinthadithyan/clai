from json import load,dump
from random import shuffle

def create_split(file_path,output_path,split_size_test=100,template=True):
    """Create splits with a given json file"""
    with open(file_path) as json_file:
        data = load(json_file)
    list_of_stat = []
    if template:
        for i in data:
            to_append = {}
            to_append["invocation"] = " ".join(i["NL"])
            to_append["cmd"] = " ".join(i["Cmd"])
            list_of_stat.append(to_append)
    else:
        for i in range(1,len(data)+1):
            list_of_stat.append(data[str(i)])
    shuffle(list_of_stat)
    train_stat = list_of_stat[:-split_size_test]
    test_stat  = list_of_stat[-split_size_test:]
    write_file_json(output_path+"train.json",train_stat)
    write_file_json(output_path+"test.json",test_stat)
    return train_stat,test_stat

def write_file_json(Filepath,Object):
    with open(Filepath, 'w') as outfile:
        dump(Object, outfile)
    return "Data Written"

def write_test_data(test_path,output_path):
    with open(test_path) as json_file:
        data = load(json_file)
    annotation = []
    for i in data:
        annotation.append(i["invocation"])
    with open(output_path+"/test.anno", 'w') as f:
        for item in annotation:
            f.write("%s\n" % item)
    return None
if __name__ == "__main__":
    create_split(r"/home/reshinth-adith/reshinth/clai/data/template/Template.json",
                r"/home/reshinth-adith/reshinth/clai/data/template/split//",
                )
    create_split(r"/home/reshinth-adith/reshinth/clai/data/raw/raw.json",
                r"/home/reshinth-adith/reshinth/clai/data/raw/split//",
                template=False)
    write_test_data(r"/home/reshinth-adith/reshinth/clai/data/raw/split/test.json",
                     r"/home/reshinth-adith/reshinth/clai/data/raw/split/")
    write_test_data(r"/home/reshinth-adith/reshinth/clai/data/template/split/test.json",
                     r"/home/reshinth-adith/reshinth/clai/data/template/split/")