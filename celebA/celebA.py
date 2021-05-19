#%%
import torch
def get_landmarks(fname='list_landmarks_celeba.txt'):
    landmarks = []
    line_number = 0
    pic_number = 0
    landmarks_names = None
    with open(fname, 'r') as f:
        for line_number, line in enumerate(f):
            if line_number == 0:
                pic_number = int(line)
            elif line_number == 1:
                landmarks_names = line.strip().split()
            else:
                marks = [int(x) for x in line.strip().split()[1:]]  # first element is filename
                landmarks.append(marks)
    return torch.tensor(landmarks)

landmarks = get_landmarks('list_landmarks_align_celeba.txt')
#%%
print(landmarks.shape)
# %%
