# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Anim

import flatbuffers
from flatbuffers.compat import import_numpy

from GFLib.Anim.CommandEntry import CommandEntryT, CommandEntry

np = import_numpy()

class CommandTrack(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CommandTrack()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsCommandTrack(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # CommandTrack
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CommandTrack
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # CommandTrack
    def FrameStart(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # CommandTrack
    def FrameLen(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # CommandTrack
    def Vec(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = CommandEntry()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # CommandTrack
    def VecLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CommandTrack
    def VecIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def CommandTrackStart(builder):
    builder.StartObject(4)

def Start(builder):
    CommandTrackStart(builder)

def CommandTrackAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    CommandTrackAddName(builder, name)

def CommandTrackAddFrameStart(builder, frameStart):
    builder.PrependUint32Slot(1, frameStart, 0)

def AddFrameStart(builder, frameStart):
    CommandTrackAddFrameStart(builder, frameStart)

def CommandTrackAddFrameLen(builder, frameLen):
    builder.PrependUint32Slot(2, frameLen, 0)

def AddFrameLen(builder, frameLen):
    CommandTrackAddFrameLen(builder, frameLen)

def CommandTrackAddVec(builder, vec):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(vec), 0)

def AddVec(builder, vec):
    CommandTrackAddVec(builder, vec)

def CommandTrackStartVecVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartVecVector(builder, numElems):
    return CommandTrackStartVecVector(builder, numElems)

def CommandTrackEnd(builder):
    return builder.EndObject()

def End(builder):
    return CommandTrackEnd(builder)

try:
    from typing import List
except:
    pass

class CommandTrackT(object):

    # CommandTrackT
    def __init__(self):
        self.name = None  # type: str
        self.frameStart = 0  # type: int
        self.frameLen = 0  # type: int
        self.vec = None  # type: List[CommandEntryT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        commandTrack = CommandTrack()
        commandTrack.Init(buf, pos)
        return cls.InitFromObj(commandTrack)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, commandTrack):
        x = CommandTrackT()
        x._UnPack(commandTrack)
        return x

    # CommandTrackT
    def _UnPack(self, commandTrack):
        if commandTrack is None:
            return
        self.name = commandTrack.Name()
        self.frameStart = commandTrack.FrameStart()
        self.frameLen = commandTrack.FrameLen()
        if not commandTrack.VecIsNone():
            self.vec = []
            for i in range(commandTrack.VecLength()):
                if commandTrack.Vec(i) is None:
                    self.vec.append(None)
                else:
                    commandEntry_ = CommandEntryT.InitFromObj(commandTrack.Vec(i))
                    self.vec.append(commandEntry_)

    # CommandTrackT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.vec is not None:
            veclist = []
            for i in range(len(self.vec)):
                veclist.append(self.vec[i].Pack(builder))
            CommandTrackStartVecVector(builder, len(self.vec))
            for i in reversed(range(len(self.vec))):
                builder.PrependUOffsetTRelative(veclist[i])
            vec = builder.EndVector()
        CommandTrackStart(builder)
        if self.name is not None:
            CommandTrackAddName(builder, name)
        CommandTrackAddFrameStart(builder, self.frameStart)
        CommandTrackAddFrameLen(builder, self.frameLen)
        if self.vec is not None:
            CommandTrackAddVec(builder, vec)
        commandTrack = CommandTrackEnd(builder)
        return commandTrack