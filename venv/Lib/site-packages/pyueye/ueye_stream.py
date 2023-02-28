#!/usr/bin/env python

# ------------------------------------------------------------------------------
#                 PyuEye - uEye API Python bindings
#
# Copyright (c) 2022 by IDS Imaging Development Systems GmbH.
# All rights reserved.
#
# PyuEye is a lean wrapper implementation of Python function objects that
# represent uEye API functions. These bindings could be used as is.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------

__author__ = "IDS Imaging Development Systems GmbH"
__copyright__ = "Copyright 2022, IDS Imaging Development Systems GmbH"
__maintainer__ = "IDS Imaging Development Systems GmbH"

import ctypes
import logging
import platform

from .dll import load_dll
from . import ueye


get_dll_file, _bind = load_dll("ueye_stream", ["ueye_stream", "ueye_stream"], "PYUEYE_STREAM_DLL_PATH")

logger = logging.getLogger(__name__)

IS_STREAM_ID = ueye.c_void_p


class IS_STREAM_PIXELFMT(ueye._CtypesEnum):
    IS_STREAM_FMT_NONE = - 1
    IS_STREAM_FMT_RGBA = 28
    IS_STREAM_FMT_BGRA = 30
    IS_STREAM_FMT_RGB24 = 2
    IS_STREAM_FMT_BGR24 = 3


IS_STREAM_FMT_NONE = IS_STREAM_PIXELFMT.IS_STREAM_FMT_NONE
IS_STREAM_FMT_RGBA = IS_STREAM_PIXELFMT.IS_STREAM_FMT_RGBA
IS_STREAM_FMT_BGRA = IS_STREAM_PIXELFMT.IS_STREAM_FMT_BGRA
IS_STREAM_FMT_RGB24 = IS_STREAM_PIXELFMT.IS_STREAM_FMT_RGB24
IS_STREAM_FMT_BGR24 = IS_STREAM_PIXELFMT.IS_STREAM_FMT_BGR24


class IS_STREAM_CODEC(ueye._CtypesEnum):
    IS_STREAM_CODEC_NONE = 0
    IS_STREAM_CODEC_H264 = 28
    IS_STREAM_CODEC_MJPEG = 8


IS_STREAM_CODEC_NONE = IS_STREAM_CODEC.IS_STREAM_CODEC_NONE
IS_STREAM_CODEC_H264 = IS_STREAM_CODEC.IS_STREAM_CODEC_H264
IS_STREAM_CODEC_MJPEG = IS_STREAM_CODEC.IS_STREAM_CODEC_MJPEG


class IS_STREAM_TYPES(ueye._CtypesEnum):
    IS_STREAM_TYPE_VIDEO = 0
    IS_STREAM_TYPE_AUDIO = 1


IS_STREAM_TYPE_VIDEO = IS_STREAM_TYPES.IS_STREAM_TYPE_VIDEO
IS_STREAM_TYPE_AUDIO = IS_STREAM_TYPES.IS_STREAM_TYPE_AUDIO


class IS_STREAM_COMMAND(ueye._CtypesEnum):
    IS_STREAM_INIT = 0
    IS_STREAM_EXIT = 1
    IS_STREAM_ADD_SESSION = 2
    IS_STREAM_REMOVE_SESSION = 3
    IS_STREAM_ADD_STREAM = 4
    IS_STREAM_SUBMIT_DATA = 5
    IS_STREAM_GET_SESSION_INFO = 6


IS_STREAM_INIT = IS_STREAM_COMMAND.IS_STREAM_INIT
IS_STREAM_EXIT = IS_STREAM_COMMAND.IS_STREAM_EXIT
IS_STREAM_ADD_SESSION = IS_STREAM_COMMAND.IS_STREAM_ADD_SESSION
IS_STREAM_REMOVE_SESSION = IS_STREAM_COMMAND.IS_STREAM_REMOVE_SESSION
IS_STREAM_ADD_STREAM = IS_STREAM_COMMAND.IS_STREAM_ADD_STREAM
IS_STREAM_SUBMIT_DATA = IS_STREAM_COMMAND.IS_STREAM_SUBMIT_DATA
IS_STREAM_GET_SESSION_INFO = IS_STREAM_COMMAND.IS_STREAM_GET_SESSION_INFO


class IS_SESSION_INFO(ueye._Structure):
    _pack_ = 8
    _fields_ = [
        ("cbSizeOfStruct", ueye.c_int),
        ("idSession", ueye.c_void_p),
        ("strName", (ueye.c_char * 20)),
        ("strInfo", (ueye.c_char * 20)),
        ("strDescription", (ueye.c_char * 20)),
        ("strURL", (ueye.c_char * 255)),
    ]

    def __init__(self, **kwargs):
        self.cbSizeOfStruct = ueye.c_int()
        self.idSession = ueye.c_void_p()
        self.strName = bytes(20)
        self.strInfo = bytes(20)
        self.strDescription = bytes(20)
        self.strURL = bytes(255)

        super(IS_SESSION_INFO, self).__init__(**kwargs)


class IS_STREAM_INFO(ueye._Structure):
    _pack_ = 8
    _fields_ = [
        ("cbSizeOfStruct", ueye.c_int),
        ("idSession", ueye.c_void_p),
        ("idStream", ueye.c_void_p),
        ("streamType", ueye.c_int),
        ("pStreamData", ueye.c_void_p),
    ]

    _map_ = {
        "streamType": IS_STREAM_TYPES
    }

    def __init__(self, **kwargs):
        self.cbSizeOfStruct = ueye.c_int()
        self.idSession = ueye.c_void_p()
        self.idStream = ueye.c_void_p()
        self.streamType = ueye.c_int()
        self.pStreamData = ueye.c_void_p()

        super(IS_STREAM_INFO, self).__init__(**kwargs)


class IS_STREAM_VIDEO_CONFIGURATION(ueye._Structure):
    _pack_ = 8
    _fields_ = [
        ("cbSizeOfStruct", ueye.c_int),
        ("srcWidth", ueye.c_int),
        ("srcHeight", ueye.c_int),
        ("dstWidth", ueye.c_int),
        ("dstHeight", ueye.c_int),
        ("srcPixelformat", ueye.c_int),
        ("dstPixelformat", ueye.c_int),
        ("srcCodec", ueye.c_int),
        ("dstCodec", ueye.c_int),
        ("framerate", ueye.c_int),
        ("bitrate", ueye.c_uint),
    ]

    _map_ = {
        "srcPixelformat": IS_STREAM_PIXELFMT,
        "dstPixelformat": IS_STREAM_PIXELFMT,
        "srcCodec": IS_STREAM_CODEC,
        "dstCodec": IS_STREAM_CODEC
    }

    def __init__(self, **kwargs):
        self.cbSizeOfStruct = ueye.c_int()
        self.srcWidth = ueye.c_int()
        self.srcHeight = ueye.c_int()
        self.dstWidth = ueye.c_int()
        self.dstHeight = ueye.c_int()
        self.srcPixelformat = ueye.c_int()
        self.dstPixelformat = ueye.c_int()
        self.srcCodec = ueye.c_int()
        self.dstCodec = ueye.c_int()
        self.framerate = ueye.c_int()
        self.bitrate = ueye.c_uint()

        super(IS_STREAM_VIDEO_CONFIGURATION, self).__init__(**kwargs)


class IS_STREAM_PAYLOAD_DATA(ueye._Structure):
    _pack_ = 8
    _fields_ = [
        ("cbSizeOfStruct", ueye.c_int),
        ("idSession", ueye.c_void_p),
        ("idStream", ueye.c_void_p),
        ("pData", ctypes.POINTER(ueye.c_ubyte)),
        ("cbSizeOfData", ueye.c_int),
    ]

    def __init__(self, **kwargs):
        self.cbSizeOfStruct = ueye.c_int()
        self.idSession = ueye.c_void_p()
        self.idStream = ueye.c_void_p()
        self.pData = ctypes.POINTER(ueye.c_ubyte)()
        self.cbSizeOfData = ueye.c_int()

        super(IS_STREAM_PAYLOAD_DATA, self).__init__(**kwargs)


IS_STREAM_NO_SUCCESS = -1
IS_STREAM_SUCCESS = 0
IS_STREAM_NOT_INITIALIZED = 1
IS_STREAM_ALREADY_INITIALIZED = 2
IS_STREAM_INVALID_PARAMETER = 3
IS_STREAM_NAME_IN_USE = 4
IS_STREAM_INVALID_ID = 5
IS_STREAM_NO_MEMORY = 6
IS_STREAM_ENCODER_ERROR = 7


_is_Stream = _bind("is_Stream", [ctypes.c_int, ctypes.c_void_p, ctypes.c_int], ctypes.c_int)


def is_Stream(nCommand, pParam, cbSizeOfParam):
    """
    :param nCommand: c_int (aka c-type: int)
    :param pParam: c_void_p - might differ depending on nCommand (aka c-type: void \*)
    :param cbSizeOfParam: c_int (aka c-type: int)
    :returns: success, or no success, that is the answer
    :raises NotImplementedError: if function could not be loaded
    """
    if _is_Stream is None:
        raise NotImplementedError()

    _nCommand = ueye._value_cast(nCommand, ctypes.c_int)
    _pParam = ueye._pointer_cast(pParam, ctypes.c_void_p)
    _cbSizeOfParam = ueye._value_cast(cbSizeOfParam, ctypes.c_int)

    ret = _is_Stream(_nCommand, _pParam, _cbSizeOfParam)

    return ret




