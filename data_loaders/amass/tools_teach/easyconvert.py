# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from data_loaders.amass.tools_teach import geometry


def nfeats_of(rottype):
    if rottype in ["rotvec", "axisangle"]:
        return 3
    elif rottype in ["rotquat", "quaternion"]:
        return 4
    elif rottype in ["rot6d", "6drot", "rotation6d"]:
        return 6
    elif rottype in ["rotmat"]:
        return 9
    else:
        return TypeError("This rotation type doesn't have features.")


def axis_angle_to(newtype, rotations):
    if newtype in ["matrix"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.axis_angle_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        return rotations
    else:
        raise NotImplementedError


def matrix_to(newtype, rotations):
    if newtype in ["matrix"]:
        return rotations
    if newtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 9))
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = geometry.matrix_to_axis_angle(rotations)
        return rotations
    else:
        raise NotImplementedError


def to_matrix(oldtype, rotations):
    if oldtype in ["matrix"]:
        return rotations
    if oldtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 3, 3))
        return rotations
    elif oldtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = geometry.quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError
