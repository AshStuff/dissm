import os
import json

def main():
    folders = [ 'DatasetSplit/PancreasDecathlon']
    files = ['train', 'test', 'val']
    for folder in folders:
        for file in files:

            cur_json = []
            with open(os.path.join(folder, file + '.txt'), 'r') as f:
                lines = f.readlines()

            for line in lines:
                cur_id = line.split()[0]
                cur_filename = 'pancreas_' + cur_id + '.obj.npz'
                cur_json.append({'path': cur_filename})

            with open(os.path.join(folder, file + '.json'), 'w') as f:
                json.dump(cur_json, f)
if __name__ == '__main__':
    main()
