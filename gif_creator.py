import cv2
import imageio

import numpy as np

from tensorboard.backend.event_processing import event_accumulator


if __name__ == "__main__":

    tb_logs_path = ""
    gif_path = ""
    images = list()
    visualize = False
    save_step = 25
    fps = 2

    event_acc = event_accumulator.EventAccumulator(tb_logs_path, size_guidance={"images": 0})
    event_acc.Reload()

    for tag in event_acc.Tags()["images"]:
        events = event_acc.Images(tag)
        for i, event in enumerate(events):
            if not (i == 0 or i % save_step == 0):
                continue
            string = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(string, cv2.IMREAD_COLOR)
            if visualize:
                cv2.imshow("Image", image)
                cv2.waitKey(0)
            images.append(image)

    imageio.mimsave(gif_path, images, fps=fps)
