import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import os
import json


# This function is used to generate vector images for the VQA task
# use_anno is used to determine whether to use the annotation to filter the generated images
def genVectorImg(pred_json_name, save_path_root, annotation, use_anno=False):
    with open(pred_json_name) as f:
        data_new = json.load(f)

    centerlines, leftlines, rightlines = [], [], []
    for lane in data_new["predictions"]['lane_segment']:
        centerlines.append(lane['centerline'])
        leftlines.append(lane['left_laneline'])
        rightlines.append(lane['right_laneline'])
    
    print('centerlines_num:', len(centerlines))
    
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]

    for index1, (cent1, left1, right1) in enumerate(zip(centerlines, leftlines, rightlines)):
        for index2, (cent2, left2, right2) in enumerate(zip(centerlines, leftlines, rightlines)):
            if index1 < index2: 
                if use_anno and (index1, index2) not in annotation:
                    continue
                fig, ax = plt.subplots(figsize=(10, 20))
                ax.set_axis_off()
                fig.patch.set_facecolor('black')

                for lanes in [leftlines, rightlines]:
                    for lane in lanes:
                        x_vals, y_vals = zip(*[( -y, x) for x, y, z in lane])
                        ax.plot(x_vals, y_vals, color=colors[0], linewidth=1.5, zorder=4)

                palette = ['green', 'blue']

                def transform_lane(lane):
                    return [( -y, x) for x, y, z in lane]

                def plot_polygon(lane1, lane2, color):
                    points = transform_lane(lane1) + list(reversed(transform_lane(lane2)))
                    polygon = patches.Polygon(points, closed=True, fill=True, color=color, zorder=5)
                    ax.add_patch(polygon)

                def plot_arrow(start, end, color):
                    arrow = mpatches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=50, lw=6, color=color, zorder=7)
                    ax.add_patch(arrow)

                mid_idx1, mid_idx2 = len(cent1) // 2, len(cent2) // 2
                cent_point1 = transform_lane([cent1[-5], cent1[-1]])
                mid_point1, mid_point2 = transform_lane([cent1[mid_idx1]])[0], transform_lane([cent2[mid_idx2]])[0]

                plot_polygon(left1, right1, palette[0])
                plot_polygon(left2, right2, palette[1])
                plot_arrow(cent_point1[0], cent_point1[1], 'white')
                plot_arrow(mid_point1, mid_point2, 'magenta')

                delta_x, delta_y = 18, 40
                vector1 = np.array([cent_point1[1][0] - mid_point1[0], cent_point1[1][1] - mid_point1[1]])
                vector2 = np.array([mid_point2[0] - mid_point1[0], mid_point2[1] - mid_point1[1]])
                delta_length = 10
                unit_vector1, unit_vector2 = vector1 / np.linalg.norm(vector1), vector2 / np.linalg.norm(vector2)
                arrow_ori = (delta_x, delta_y)
                plot_arrow(arrow_ori, (delta_x + unit_vector1[0] * delta_length, delta_y + unit_vector1[1] * delta_length), 'white')
                plot_arrow(arrow_ori, (delta_x + unit_vector2[0] * delta_length, delta_y + unit_vector2[1] * delta_length), 'magenta')
                
                name = f"{save_path_root}/{os.path.splitext(os.path.basename(pred_json_name))[0]}-{index1}-{index2}.png"
                plt.savefig(name, bbox_inches='tight', pad_inches=0, facecolor='black')
                plt.close(fig)
    return

def read_txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=3)
                if len(parts) >= 3:
                    main_key = (parts[0], "-".join(parts[1].split('-')[:-2]))
                    sub_key = tuple(map(int, parts[1].split('-')[-2:]))
                    value = parts[3] if len(parts) > 3 else ""
                    data_dict.setdefault(main_key, {})[sub_key] = value
    return data_dict

def main():
    pred_path = './dataset/output_json/'    
    save_path_root = './dataset/VQA/vector/'
    anno_path = "./dataset/VQA_annotation.txt"
    annos = read_txt_to_dict(anno_path)
    
    for i in range(10000, 10150):
        pred_info_path = f"{pred_path}{str(i).zfill(5)}/info"
        for b in os.listdir(pred_info_path):
            key = (str(i).zfill(5), b.split('.')[0])
            if key not in annos:
                continue
            annotation = annos[key]
            save_path = f"{save_path_root}/{str(i).zfill(5)}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            genVectorImg(f"{pred_info_path}/{b}", save_path, annotation, use_anno=True)      
        print(f'The Scene {i} done')
    print("Done!")

if __name__ == '__main__':
    main()
