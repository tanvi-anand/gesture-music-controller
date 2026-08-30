import cv2

class CameraStream:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise IOError("Could not open webcam. Check that it is connected and not in use by another program.")

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def release(self):
        self.camera.release()