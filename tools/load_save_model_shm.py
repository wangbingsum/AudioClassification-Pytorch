# -*- coding: utf-8 -*-
import os
import sys
import ctypes
import onnx
from io import BytesIO


class shmctrl:
    def __init__(self):
        self._handle = ctypes.CDLL(
            "librt.so", use_errno=True, use_last_error=True)
        shm_key_t = ctypes.c_int
        # key_t ftok (const char *__pathname, int __proj_id);
        self.ftok = self._handle.ftok
        self.ftok.restype = shm_key_t
        self.ftok.argtypes = [ctypes.c_char_p, ctypes.c_int]
        # int shmget(key_t key, size_t size, int shmflg);
        self.shmget = self._handle.shmget
        self.shmget.restype = ctypes.c_int
        self.shmget.argtypes = (shm_key_t, ctypes.c_size_t, ctypes.c_int)
        # void* shmat(int shmid, const void *shmaddr, int shmflg);
        self.shmat = self._handle.shmat
        self.shmat.restype = ctypes.c_void_p
        self.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        # int shmdt(const void *shmaddr);
        self.shmdt = self._handle.shmdt
        self.shmdt.restype = ctypes.c_int
        self.shmdt.argtypes = (ctypes.c_void_p,)
        # int shmctl(int shmid, int cmd, struct shmid_ds *buf);
        self.shmctl = self._handle.shmctl
        self.shmctl.restype = ctypes.c_int
        self.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        # void* memcpy( void *dest, const void *src, size_t count );
        self.memcpy = self._handle.memcpy
        self.memcpy.restype = ctypes.c_void_p
        self.memcpy.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            ctypes.FreeLibrary(self._handle)


def loadonnxmodel_shm(model_size,file,verbose):
    shm = shmctrl()

    # ======================
    # process share memory1
    # ======================
    shm1_size = model_size # shared memory1 passed by argument
    shm1_key = shm.ftok(ctypes.c_char_p(file.encode()), 0)
    shm1_id = shm.shmget(shm1_key, 0, 0o666)

    if shm1_id < 0:
        print("failed to got share memory1")
        return -1
    else:
        shm1 = shm.shmat(shm1_id, None, 0)

    # read onnx model from share memory1
    shm1_py = ctypes.string_at(shm1, shm1_size)
    f = BytesIO(shm1_py)
    model_proto_from_binary_stream = onnx.load(f, onnx.ModelProto)

    # ======================
    # detach share memory
    # ======================
    shm.shmdt(shm1)
    if verbose >= 2:
        print("1.onnx model is loaded from share memory")
    return model_proto_from_binary_stream


def loadonnxmodel_shm2(model_size2,file,verbose):
    shm = shmctrl()

    # ======================
    # process share memory2
    # ======================
    shm2_size = model_size2 # shared memory2 passed by argument
    shm2_key = shm.ftok(ctypes.c_char_p(file.encode()), 1)
    shm2_id = shm.shmget(shm2_key, 0, 0o666)

    if shm2_id < 0:
        print("failed to got share memory2")
        return -1
    else:
        shm2 = shm.shmat(shm2_id, None, 0)

    # read onnx model from share memory2
    shm2_py = ctypes.string_at(shm2, shm2_size)
    f = BytesIO(shm2_py)
    model_proto_from_binary_stream = onnx.load(f, onnx.ModelProto)

    # ======================
    # detach share memory
    # ======================
    shm.shmdt(shm2)

    if verbose >= 2:
        print("1.onnx model is loaded from share memory")
    return model_proto_from_binary_stream


def saveonnxmodel_shm(onnxmodel, file, verbose):
    try:
        shm = shmctrl()
        # ======================
        # process share memory2
        # ======================
        shm2_size = onnxmodel.ByteSize()
        shm2_key = shm.ftok(ctypes.c_char_p(file.encode()), 1)
        shm2_id = shm.shmget(shm2_key, shm2_size, 0o1666)

        if shm2_id < 0:
            print("failed to create share memory2.")
            return -1
        else:
            shm2 = shm.shmat(shm2_id, None, 0)

        ## save onnxmodel to share memory2
        shm.memcpy(shm2, onnx._serialize(onnxmodel), shm2_size)

        # ======================
        # detach share memory
        # ======================
        shm.shmdt(shm2)
        if verbose >= 2:
            print("2.The onnx model is saved successfully and has been saved to share memory")
    except Exception as e:
        print("2.There is a problem with the onnx model and it was not saved successfully:\n",e)


def saveonnxmodel_shm2(onnxmodel, file, verbose):
    try:
        shm = shmctrl()
        # ======================
        # process share memory3
        # ======================
        shm3_size = onnxmodel.ByteSize()
        shm3_key = shm.ftok(ctypes.c_char_p(file.encode()), 2)
        shm3_id = shm.shmget(shm3_key, shm3_size, 0o1666)

        if shm3_id < 0:
            print("failed to create share memory2.")
            return -1
        else:
            shm3 = shm.shmat(shm3_id, None, 0)

        ## save onnxmodel to share memory3
        shm.memcpy(shm3, onnx._serialize(onnxmodel), shm3_size)

        # ======================
        # detach share memory
        # ======================
        shm.shmdt(shm3)
        if verbose >= 2:
            print("2.The onnx model is saved successfully and has been saved to share memory")
    except Exception as e:
        print("2.There is a problem with the onnx model and it was not saved successfully:\n",e)