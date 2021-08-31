#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import unicode_literals, print_function

import os
import ROOT

class IteratorItem:
    """
    A simple class to access items returned by Iterator.next().

    Usage example: see below.
    """

    def __init__(self, data, filename, treename, index=-1):
        """
        Initialize a new iterator item.

        Args:
            data (dict): Data dictionary associated with this item.
            filename (string): Name of the originating ROOT file.
            treename (string): Name of the originating ROOT tree.
            index (int): Index of the originating tree item.
        """
        assert(type(data) == dict)
        assert(type(filename) == str)
        assert(type(treename) == str)
        assert(type(index) == int)

        self.data = data
        self.filename = filename
        self.treename = treename
        self.index = index

    def __str__(self):
        return "%s:%s:%d" % (os.path.split(self.filename)[-1], self.treename, self.index)

    def __len__(self):
        return len(self.data)

    def __dir__(self):
        return self.data.keys()

    def __getattr__(self, name):
        assert(type(name) == str)

        if not name in self.data:
            raise KeyError("IteratorItem does not contain a field named '%s'" % name)

        return self.data[name]

class Iterator:
    """
    A simple class for reading Kassiopeia output.

    The class defines an iterator that can be used to access tree entries
    in sequential order. This is typically more resource-effective than
    loading the entire ROOT tree into memory before processing the data.

    Usage example:
        import KassiopeiaReader

        reader = KassiopeiaReader.Iterator("QuadrupoleTrapSimulation.root")

        print("List of tree names:", dir(reader))

        reader.loadTree('component_step_world')

        print("List of tree objects:", dir(reader))
        print("Total number of tracks:", len(reader.getTracks('TRACK_INDEX')))
        print("Total number of steps:", len(reader))  # do not load all steps here!

        first_step = reader.getTracks('FIRST_STEP_INDEX')[0]  # first step of first track
        last_step = reader.getTracks('LAST_STEP_INDEX')[0]  # last step of first track

        num_steps = last_step - first_step + 1

        print("Number of steps that will be read:", num_steps)

        reader.select('step_id', 'kinetic_energy')  # only read two fields from tree

        sum_energy = 0.
        for item in iter(reader):
            sum_energy += item.kinetic_energy
            if item.step_id >= last_step:
                break

        print("Last step that was read:", step_id)
        print("Average energy in 1st track:", sum_energy / num_steps)

        reader.closeFile()
"""

    def __init__(self, filename="", treename=""):
        """
        Initialize reader and open ROOT file, if given.

        One can open multiple files sequentially with the same reader
        object by the openFile() method.

        The ROOT tree must be loaded by the loadTree() method before
        accessing the iterator. Alternatively one can load the entire
        tree by the readTree() method, although this is not advised for
        large trees since iterating is typically more efficient.

        Args:
            filename (string): The name of the ROOT file to be opened.
            treename (string): The name of the ROOT tree to be loaded.
        """
        assert(type(filename) == str)
        assert(type(treename) == str)

        self.file = None
        self.closeFile()  # initialize

        if filename:
            self.openFile(filename)

            if treename:
                self.loadTree(treename)

    def __str__(self):
        if self.file:
            if self.tree:
                return "%s:%s" % (self.file.GetName(), self.tree.GetName())
            else:
                return "%s" % (self.file.GetName())
        else:
            return "(empty)"

    def __dir__(self):
        if self.file:
            if self.tree:
                return self.leaves.keys()
            else:
                return self.treenames
        else:
            return []

    def __len__(self):
        return int(self.nev)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def openFile(self, filename):
        """
        Open ROOT file for reading.

        Args:
            filename (string): The name of the ROOT file.
        """
        assert(type(filename) == str)

        if self.file:
            self.closeFile()

        if not os.path.exists(filename):
            raise IOError("File '%s' does not exist" % (filename))

        self.file = ROOT.TFile(filename, 'read')
        if self.file.IsZombie():
            self.closeFile()
            raise IOError("TFile '%s' could not be opened" % (filename))

        forest = self.file.GetListOfKeys()

        self.treenames = []
        for i in range(forest.GetEntries()):
            tree = forest.At(i)
            name = tree.GetName()
            if not name.endswith('_DATA'):
                continue

            self.treenames.append(name[:-5])  # strip off '_DATA'

    def closeFile(self):
        """
        Close the previously opened ROOT file.
        """
        if self.file:
            if self.file.IsOpen():
                self.file.Close()

        self.file = None
        self.iev = 0
        self.nev = 0

        self.tree = None
        self.treenames = []
        self.leaves = {}

    def getRuns(self, fieldname=""):
        assert(type(fieldname) == str)

        return self.getTree('RUN_DATA', fieldname)

    def getEvents(self, fieldname=""):
        assert(type(fieldname) == str)

        return self.getTree('EVENT_DATA', fieldname)

    def getTracks(self, fieldname=""):
        assert(type(fieldname) == str)

        return self.getTree('TRACK_DATA', fieldname)

    def getSteps(self, fieldname=""):
        assert(type(fieldname) == str)

        return self.getTree('STEP_DATA', fieldname)

    def getTree(self, treename='RUN_DATA', fieldname=""):
        """
        Return the entire contents of the given tree.

        Note:
            For accessing the tree contents, the iterator approach
            provides a faster method.

        Args:
            treename (string): The name of a tree in the ROOT file.

        Returns:
            dict: Full contents of the tree, like
                    { 'column1': [ values1 ], 'column2': [ values2 ], ... }.
        """
        assert(type(treename) == str)
        assert(type(fieldname) == str)

        if not self.file:
            raise RuntimeError("A file must be opened before accessing a tree.")

        if not (treename.endswith('_DATA') or treename.endswith('_PRESENCE') or treename.endswith('_STRUCTURE')):
            treename += '_DATA'

        tree = self.file.Get(treename)
        foliage = tree.GetListOfLeaves()

        leaves = {}
        for i in range(foliage.GetEntries()):
            leaf = foliage.At(i)
            name = leaf.GetName()
            leaves[name] = leaf

        data = {}
        for j in range(tree.GetEntries()):
            tree.GetEntry(j)

            for name, leaf in leaves.items():
                if fieldname and name != fieldname:
                    continue

                if not name in data:
                    data[name] = []
                data[name].append( eval("tree." + name) )

        if fieldname:
            return data[fieldname]
        else:
            return data

    def loadTree(self, treename='output_track_world'):
        """
        Prepare reading from the given tree and load column names.

        After loading, the tree contents can be read using the
        iterator structure of this class.

        Note:
            The ``select`` function allows to narrow down the
            number of values that is read at each iteration.

        Args:
            treename (string): The name of a tree in the ROOT file.

        Returns:
            int: The number of tree entries.
        """
        assert(type(treename) == str)

        if not self.file:
            raise RuntimeError("A file must be opened before accessing a tree.")

        if not (treename.endswith('_DATA') or treename.endswith('_PRESENCE') or treename.endswith('_STRUCTURE')):
            treename += '_DATA'

        try:
            tree = self.file.Get(treename)
            foliage = tree.GetListOfLeaves()
        except AttributeError:
            raise AttributeError("TFile '%s' has no tree named '%s'" % (self.file.GetName(), treename))

        self.tree = tree
        self.leaves = {}
        for i in range(foliage.GetEntries()):
            leaf = foliage.At(i)
            name = leaf.GetName()
            self.leaves[name] = leaf

        self.iev = 0
        self.nev = self.tree.GetEntries()

    def reset(self):
        """
        Reset the iterator to start at the first tree entry.
        """
        self.iev = 0

    def select(self, *fields):
        """
        Select a given list of columns to be read at each iteration.

        Args:
            fields (string): One or more names of fields to read while iterating.
        """
        assert(type(fields) == tuple)

        leaves = {}
        for name in fields:
            assert(type(name) == str)

            if name in self.leaves.keys():
                leaves[name] = self.leaves[name]

        self.leaves = leaves

    def next(self):
        """
        Iterate through the tree that was loaded before.

        Returns:
            (IteratorItem): Contents of the tree at the current entry
                    as a dictionary-like object.
        """
        if not self.file:
            raise RuntimeError("A file must be opened before accessing a tree.")

        if self.iev >= self.nev:
            self.reset()
            raise StopIteration()

        self.tree.GetEntry(self.iev)
        self.iev += 1

        data = {}
        for name, leaf in self.leaves.items():
            data[name] = eval("self.tree." + name)

        return IteratorItem(data, self.file.GetName(), self.tree.GetName(), self.iev-1)
