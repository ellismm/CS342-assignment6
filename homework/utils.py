import torch
import torch.nn.functional as F
import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'

DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def quat_rot(r, v, inverse=False):
    inv = 1 - 2 * float(inverse)
    return np.array([(1 - 2 * (r[(i + 1) % 3] ** 2 + r[(i + 2) % 3] ** 2)) * v[i] +
                     2 * (r[i] * r[(i + 1) % 3] - r[(i + 2) % 3] * r[3] * inv) * v[(i + 1) % 3] +
                     2 * (r[i] * r[(i + 2) % 3] + r[(i + 1) % 3] * r[3] * inv) * v[(i + 2) % 3] for i in range(3)])


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    def _to_world(self, aim_point_image, proj, view, height=0):
        x, y, W, H = *aim_point_image, self.config.screen_width, self.config.screen_height
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
        t = (height-x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1-t) * x0

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_kart(x, kart):
        return quat_rot(kart.rotation, x - kart.location, True)

    def _to_image(self, x, proj, view):
        W, H = self.config.screen_width, self.config.screen_height
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        do_render = verbose or planner is not None or data_callback is not None
        if self.k is not None and self.k.config.track == track and do_render == self.k.config.render:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, render=do_render, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):

            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            if data_callback is not None:
                aim_point_world = self._point_on_track(kart.distance_down_track + 20, track)
                ap = self._to_image(aim_point_world, proj, view)
                if 0 <= ap[0] < self.config.screen_width and 0 <= ap[1] < self.config.screen_height:
                    data_callback(t, np.array(self.k.render_data[0].image), ap)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0)
                aim_point_world = self._to_world(aim_point_image, proj, view, kart.location[1])
            else:
                aim_point_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)

            aim_point_car = self._to_kart(aim_point_world, kart)
            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_car, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                ax.add_artist(plt.Circle(self._to_image(kart.location, proj, view), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(self._to_image(aim_point_world, proj, view), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(self._to_image(ap, proj, view), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


if __name__ == '__main__':
    from .controller import control
    from argparse import ArgumentParser
    from os import makedirs


    def noisy_control(aim_pt, vel):
        return control(aim_pt + np.random.randn(*aim_pt.shape) * aim_noise,
                       vel + np.random.randn() * vel_noise)


    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=10000, type=int)
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int)
    parser.add_argument('--aim_noise', default=5, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0


        def collect(_, im, pt):
            from PIL import Image
            from os import path
            global n
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    f.write('%0.1f,%0.1f' % tuple(pt))
            n += 1


        while n < args.steps_per_track:
            step = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            print(step)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
