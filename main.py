from deepforest import main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
model = main.deepforest()
model.use_release()

dir_path = os.getcwd()+"/frames"

os.chdir(dir_path)

for file in os.listdir():
    if(file.endswith(".jpeg")):
        file_path = f"{dir_path}/{file}"
        img = model.predict_image(path=file_path,return_plot=True)
        # plt.savefig(f"plot_{file}")
        plt.imshow(img[:,:,::-1])