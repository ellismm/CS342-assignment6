import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in local coordinate frame
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    action.acceleration = 1

    direction = aim_point[0]
    height = aim_point[1]
    distance = aim_point[2]

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    direction = aim_point[0].item()
    if(-1 < direction > 1):
        action.drift = True
    if(current_vel > 20):
        action.brake = True
        action.acceleration = 0
    else:
        action.brake = False
        
    if(direction > 1):
        direction = 1
    if(direction < -1):
        direction = -1
    if(.01 < direction > -.01) and distance > 15:
        action.nitro = True
    else:
        action.nitro = False

    # print()
    action.steer = direction

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
