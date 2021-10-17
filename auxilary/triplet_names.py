import os
from pathlib import Path

def main():
    input_dir = 'data/cv-counter'
    removed = [1,2,90,45,50,52,57,70]
    with open('./triplets_train','w') as train_file:
        for i in range(1,111):
            print('hi')
            if (not i in removed):
                name = f'{i:03}'
                anchor = f'{input_dir}/{name}.3.wav'
                positive = f'{input_dir}/{name}.3.wav'
                negative = f'{input_dir}/{name}.0.wav'

                train_file.write(f'{anchor}\t{positive}\t{negative}\n')
                

if __name__ == "__main__":
    main()
