import os
import cv2
import numpy as np

from pathlib import Path
from numpy import ndarray
from items import PanoItem


def xyz_to_lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat_to_xy(lonlat, shape):
    x = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    y = (lonlat[..., 1:] / np.pi + 0.5) * (shape[0] - 1)
    lst = [x, y]
    out = np.concatenate(lst, axis=-1)

    return out


class PanoVisualizer:
    _HEIGHT: int = None
    _WIDTH: int = None
    _PANO: ndarray = None
    _HEADING: float = None

    _IMAGE_STORE: str = os.getenv("IMAGE_STORE")
    _PANO_MODE: str = os.getenv("PANO_MODE")

    @classmethod
    @property
    def PANO(cls) -> ndarray:
        return cls._PANO

    @classmethod
    def set_pano(cls, image: str):
        if not Path(image).is_absolute():
            image = (Path(cls._IMAGE_STORE) / image).as_posix()
        cls._PANO = cv2.cvtColor(
            cv2.imread(image, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )  # R G B
        cls._HEIGHT, cls._WIDTH = cls._PANO.shape[:2]

    @classmethod
    def set_heading(cls, heading):
        if cls._PANO_MODE == "google":
            cls._HEADING = heading
        else:
            cls._HEADING = heading - 90

    @classmethod
    def _plot_view(cls, xy: ndarray) -> ndarray:
        overlay = cls._PANO.copy()
        if abs((xy[0, -1] - xy[0, 0])[0].item()) > cls._WIDTH / 2:
            corners1 = np.array([
                [0, xy[0, -1][1].item()],
                xy[0, -1],
                xy[-1, -1],
                [0, xy[-1, -1][1].item()]
            ], np.int32)

            corners2 = np.array([
                [cls._WIDTH, xy[0, 1][1].item()],
                xy[0, 0],
                xy[-1, 0],
                [cls._WIDTH, xy[-1, 0][1].item()]
            ], np.int32)

            cv2.polylines(overlay, [corners1, corners2], isClosed=True, color=(0, 255, 0), thickness=20)

        else:
            corners = np.array([
                xy[0, 0],
                xy[0, -1],
                xy[-1, -1],
                xy[-1, 0]
            ], np.int32)
            cv2.polylines(overlay, [corners], isClosed=True, color=(0, 255, 0), thickness=20)

        return overlay

    @classmethod
    def get_perspective(cls,
                        fov: int = 120,
                        theta: float = 0.0,
                        phi: int = 0,
                        height: int = 512,
                        width: int = 1024) -> PanoItem:
        """
        :param fov:
        :param theta: THETA is left/right angle
        :param phi: PHI is up/down angle, both in degree
        :param height:
        :param width:
        :return:
        """

        f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        k = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        k_inv = np.linalg.inv(k)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ k_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        r1, _ = cv2.Rodrigues(y_axis * np.radians(theta - cls._HEADING))
        r2, _ = cv2.Rodrigues(np.dot(r1, x_axis) * np.radians(phi))
        r = r2 @ r1
        xyz = xyz @ r.T
        lonlat = xyz_to_lonlat(xyz)
        xy = lonlat_to_xy(lonlat, shape=cls._PANO.shape).astype(np.float32)
        perspective = cv2.remap(cls._PANO, xy[..., 0], xy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return PanoItem(
            perspective,
            cls._plot_view(xy)
        )


def test_pano_visualization():
    # 设置图像路径
    pano_image_path = "/home/ubuntu/city-walker/images/London_7/00000003.jpg"
    
    # 设置图像
    PanoVisualizer.set_pano(pano_image_path)
    
    # 设置视角（这些参数可以根据需要调整）
    fov = 120  # 视场角度
    theta = 0.0  # 左右角度
    phi = 0  # 上下角度
    height = 512  # 高度
    width = 1024  # 宽度

    # 确保 _HEADING 已设置为一个默认值（例如0）
    PanoVisualizer.set_heading(0)  # 或者根据需要设置为其他角度
    
    # 获取切割后的图像
    pano_item = PanoVisualizer.get_perspective(fov=fov, theta=theta, phi=phi, height=height, width=width)
    
    # 获取切割后的图像并显示
    perspective_image = pano_item.perspective  # 假设 PanoItem 中使用 'perspective' 而非 'pano'
    view_image = pano_item.view
    
    # 显示图像（你可以根据自己的需求选择显示或保存）
    cv2.imshow("Perspective Image", perspective_image)
    cv2.imshow("View Image", view_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行测试
test_pano_visualization()

