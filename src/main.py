import argparse
from road_flat_eval import *
from gen_testing_road_dd_image import *
from integral_grad import *


def main():
    parser = argparse.ArgumentParser(description='Terrain Experiments')
    parser.add_argument('--experiment_type', action='store', type=str, default='RoadFlatEval',
                        help='Street map database, RoadFlatEval|GenTestingRoadDDImage|IntegralGrad')
    parser.add_argument('--dem_map_path', action='store', type=str,
                        default='D:/Codes/python/pytorch-CycleGAN-and-pix2pix/datasets/strtdem/trainA',
                        help='DEM map path')
    parser.add_argument('--dem_map_name', action='store', type=str,
                        default='D:/Codes/python/pytorch-CycleGAN-and-pix2pix/datasets/strtdem/trainA/N30.300E115.240N30.310E115.250.png',
                        help='DEM map name')
    parser.add_argument('--street_map_path', action='store', type=str,
                        default='D:/Codes/python/pytorch-CycleGAN-and-pix2pix/datasets/strtdem/trainB',
                        help='Street map export path')
    parser.add_argument('--road_perpendicular_line_path', action='store', type=str,
                        default='D:/Codes/python/pytorch-CycleGAN-and-pix2pix/datasets/strtdem/Tmp/road_perpendicular_line',
                        help='Road perpendicular line export path')
    parser.add_argument('--tmp', action='store', type=str, default='',
                        help='Tmp parameters')
    arg = parser.parse_args()

    if arg.experiment_type == 'RoadFlatEval':
        worker = RoadFlatEval(arg.dem_map_path,
                                      arg.road_perpendicular_line_path)
        worker.eval()
    elif arg.experiment_type == 'GenTestingRoadDDImage':
        worker = GenTestingRoadDDImage(arg.dem_map_path,
                                       arg.road_perpendicular_line_path)
        worker.work()
    elif arg.experiment_type == 'IntegralGrad':
        worker = IntegralGrad(arg.dem_map_name)
        worker.work()
    else:
        pass

    return


if __name__ == '__main__':
    main()
