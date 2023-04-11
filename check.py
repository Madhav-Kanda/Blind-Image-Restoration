# import cv2
# import os

# #/home/project_3/madhav/Deblur_Self/version3/GOPRO_Large/test/GOPR0410_11_00
# blur_path = '/home/project_3/madhav/Deblur_Self/version3/GOPRO_Large/test/GOPR0410_11_00/blur/'
# sharp_path = '/home/project_3/madhav/Deblur_Self/version3/GOPRO_Large/test/GOPR0410_11_00/sharp/'
# save_path = '/home/project_3/madhav/Deblur_Self/version3/test_gopro/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)


# blur_files = os.listdir(blur_path)
# sharp_files = os.listdir(sharp_path)


# blur_files.sort()
# sharp_files.sort()


# for i in range(len(blur_files)):

#     blur_img = cv2.imread(blur_path + blur_files[i])
#     blur_img = cv2.resize(blur_img, (0,0), fx=0.5, fy=0.5)


#     sharp_img = cv2.imread(sharp_path + sharp_files[i])
#     sharp_img = cv2.resize(sharp_img, (0,0), fx=0.5, fy=0.5)

#     concat_img = cv2.hconcat([sharp_img, blur_img])
#     cv2.imwrite(os.path.join(save_path, f"{i+401}.png"), concat_img)
# print(f'ALL {i+401} Saved!')

# import cv2
# import os
# import numpy as np

# dir_path = "/home/project_3/madhav/Deblur_Self/version3/train_gopro/"

# for filename in os.listdir(dir_path):
#     img = cv2.imread(os.path.join(dir_path, filename))

#     height, width = img.shape[:2]
#     width_cutoff = width // 2
#     left_img = img[:, :width_cutoff]
#     right_img = img[:, width_cutoff:]

#     swapped_img = np.concatenate((right_img, left_img), axis=1)

#     output_path = os.path.join(dir_path, f"{filename}")
#     cv2.imwrite(output_path, swapped_img)
# print('All Done!')


import cv2
import os
import numpy as np

dir_path = "/home/project_3/madhav/Deblur_Self/version3/test_gopro"

for filename in os.listdir(dir_path):
    img = cv2.imread(os.path.join(dir_path, filename))

    height, width = img.shape[:2]
    width_cutoff = width // 2
    left_img = img[:, :width_cutoff]
    right_img = img[:, width_cutoff:]

    patch_size = (256, 256)
    sharp_patch = left_img[
        height // 2 - patch_size[0] // 2 : height // 2 + patch_size[0] // 2,
        width_cutoff // 2 - patch_size[1] // 2 : width_cutoff // 2 + patch_size[1] // 2,
    ]
    blur_patch = right_img[
        height // 2 - patch_size[0] // 2 : height // 2 + patch_size[0] // 2,
        width_cutoff // 2 - patch_size[1] // 2 : width_cutoff // 2 + patch_size[1] // 2,
    ]

    combined_patch = np.hstack((sharp_patch, blur_patch))
    output_path = os.path.join(dir_path, f"{filename}")
    cv2.imwrite(output_path, combined_patch)

print("All Done!")
