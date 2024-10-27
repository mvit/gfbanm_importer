# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Anim

import flatbuffers
from flatbuffers.compat import import_numpy

from GFLib.Anim.Vec3 import Vec3, Vec3T

np = import_numpy()

class Framed16VectorTrack(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Framed16VectorTrack()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFramed16VectorTrack(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Framed16VectorTrack
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Framed16VectorTrack
    def Frames(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # Framed16VectorTrack
    def FramesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint16Flags, o)
        return 0

    # Framed16VectorTrack
    def FramesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Framed16VectorTrack
    def FramesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Framed16VectorTrack
    def Co(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 12
            obj = Vec3()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Framed16VectorTrack
    def CoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Framed16VectorTrack
    def CoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def Framed16VectorTrackStart(builder):
    builder.StartObject(2)

def Start(builder):
    Framed16VectorTrackStart(builder)

def Framed16VectorTrackAddFrames(builder, frames):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(frames), 0)

def AddFrames(builder, frames):
    Framed16VectorTrackAddFrames(builder, frames)

def Framed16VectorTrackStartFramesVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def StartFramesVector(builder, numElems):
    return Framed16VectorTrackStartFramesVector(builder, numElems)

def Framed16VectorTrackAddCo(builder, co):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(co), 0)

def AddCo(builder, co):
    Framed16VectorTrackAddCo(builder, co)

def Framed16VectorTrackStartCoVector(builder, numElems):
    return builder.StartVector(12, numElems, 4)

def StartCoVector(builder, numElems):
    return Framed16VectorTrackStartCoVector(builder, numElems)

def Framed16VectorTrackEnd(builder):
    return builder.EndObject()

def End(builder):
    return Framed16VectorTrackEnd(builder)

try:
    from typing import List
except:
    pass

class Framed16VectorTrackT(object):

    # Framed16VectorTrackT
    def __init__(self):
        self.frames = None  # type: List[int]
        self.co = None  # type: List[Vec3T]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        framed16VectorTrack = Framed16VectorTrack()
        framed16VectorTrack.Init(buf, pos)
        return cls.InitFromObj(framed16VectorTrack)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, framed16VectorTrack):
        x = Framed16VectorTrackT()
        x._UnPack(framed16VectorTrack)
        return x

    # Framed16VectorTrackT
    def _UnPack(self, framed16VectorTrack):
        if framed16VectorTrack is None:
            return
        if not framed16VectorTrack.FramesIsNone():
            if np is None:
                self.frames = []
                for i in range(framed16VectorTrack.FramesLength()):
                    self.frames.append(framed16VectorTrack.Frames(i))
            else:
                self.frames = framed16VectorTrack.FramesAsNumpy()
        if not framed16VectorTrack.CoIsNone():
            self.co = []
            for i in range(framed16VectorTrack.CoLength()):
                if framed16VectorTrack.Co(i) is None:
                    self.co.append(None)
                else:
                    vec3_ = Vec3T.InitFromObj(framed16VectorTrack.Co(i))
                    self.co.append(vec3_)

    # Framed16VectorTrackT
    def Pack(self, builder):
        if self.frames is not None:
            if np is not None and type(self.frames) is np.ndarray:
                frames = builder.CreateNumpyVector(self.frames)
            else:
                Framed16VectorTrackStartFramesVector(builder, len(self.frames))
                for i in reversed(range(len(self.frames))):
                    builder.PrependUint16(self.frames[i])
                frames = builder.EndVector()
        if self.co is not None:
            Framed16VectorTrackStartCoVector(builder, len(self.co))
            for i in reversed(range(len(self.co))):
                self.co[i].Pack(builder)
            co = builder.EndVector()
        Framed16VectorTrackStart(builder)
        if self.frames is not None:
            Framed16VectorTrackAddFrames(builder, frames)
        if self.co is not None:
            Framed16VectorTrackAddCo(builder, co)
        framed16VectorTrack = Framed16VectorTrackEnd(builder)
        return framed16VectorTrack