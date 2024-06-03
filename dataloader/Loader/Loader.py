import logging
import fnmatch
import os
from typing import Annotated, Optional
import time
import sys
import numpy as np
from typing import Tuple

import vtk
import requests
import io

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import h5py

class Loader(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AC3T LOADER") 
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "00_AC3T")]
        self.parent.dependencies = []  
        self.parent.contributors = ["Ivo Herzig (ZHAW)"]  
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
        Varian HDF volume loader.
        Loads volumes directly from server.
        Run model inference and loads results directly from GPU server """)

        self.parent.acknowledgementText = _("""
        InnoSuisse funding by XXX
        """)

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)

class LoaderFileReader:

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "Image from HDF File"

    def fileType(self):
        return "HDFImageFile"

    def extensions(self):
        return ["hdf file (*.hdf)"]

    def canLoadFile(self, path):
        ext = path[-4:].lower()
        return ext == ".hdf"

    def load(self, properties):

        file = properties['fileName']

        nodes = self.load_from_stream(file, properties['fileName'])
        self.parent.loadedNodes = [n.GetID() for n in nodes]

        return True

    def create_vol_node(self, vol: np.ndarray, vol_res: tuple[float, float, float], node_name: str, is_labelmap: bool = False ):

        directions = [[-1,0,0], [0,-1,0], [0,0,-1]] # IJK to RAS directions to load varian hdf
        if is_labelmap:
            vol_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', slicer.mrmlScene.GenerateUniqueName(node_name))
        else:
            vol_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', slicer.mrmlScene.GenerateUniqueName(node_name))

        slicer.util.updateVolumeFromArray(vol_node, vol)
        vol_node.SetSpacing(vol_res)
        vol_node.SetIJKToRASDirections(directions)
        vol_node.CreateDefaultDisplayNodes()

        return vol_node

    def load_from_stream(self, file, node_name: str, resolution_attr:str = 'Resolution'):

        result_nodes = []

        try:
            with h5py.File(file, mode='r') as hdf:
        
                proj_node = None
                if 'Projection00001' in hdf.keys():

                    proj = []
                    for name in sorted(fnmatch.filter(hdf.keys(), 'Projection*')):
                        proj.append(hdf[name]['AttenuationImage1'][()])

                    proj_res = (hdf[name]['AcqParams'].attrs['ImagerResX'],
                                hdf[name]['AcqParams'].attrs['ImagerResY'])
                    proj = np.stack(proj, axis = 0)

                    proj_node = self.create_vol_node(proj, (proj_res[0], proj_res[1], 1.0), f'Proj_{node_name}')
                    slicer.util.setSliceViewerLayers(background = proj_node)
                    slicer.util.resetSliceViews()
                    result_nodes.append(proj_node)


                vol_sequence = None
                vol_node = None
                if 'Volume' in hdf.keys():

                    vol_data = hdf['Volume'][()]
                    vol_resolution = hdf['Volume'].attrs[resolution_attr]

                    if len(vol_data.shape) == 3:

                        vol_node = self.create_vol_node(vol_data, vol_resolution, node_name)
                        result_nodes.append(vol_node)

                    elif len(vol_data.shape) == 4:

                        print(f'create sequence from 4D  vol...')
                        vol_sequence = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', f'Vol_Seq_{node_name}')
                        for i in range(vol_data.shape[0]):
                            vol_node = self.create_vol_node(vol_data[i,:,:,:], vol_resolution, f'{node_name}_{i}')
                            vol_sequence.SetDataNodeAtValue(vol_node, str(i))
                            slicer.mrmlScene.RemoveNode(vol_node)
                            result_nodes.append(vol_sequence)


                    else:
                        raise RuntimeError(f'only 3D and 4D volumes are support got: {len(vol_data.shape)}D (shape: {vol_data.shape})')

                seg_sequence = None
                if 'TotalSegmentator' in hdf.keys():
                    segmentation = hdf['TotalSegmentator'][()]
                    vol_resolution = hdf['TotalSegmentator'].attrs[resolution_attr]

                    labels = hdf['TotalSegmentator'].attrs['labels']
                    ids = hdf['TotalSegmentator'].attrs['ids']

                    color_table_node = slicer.vtkMRMLColorTableNode() # slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode", "total")
                    color_table_node.SetTypeToUser()
                    color_table_node.SetNumberOfColors(len(labels)+1)
                    #slicer.mrmlScene.AddNode(color_table_node); color_table_node.UnRegister(None)
                    
                    random_colors = slicer.mrmlScene.GetNodeByID('vtkMRMLColorTableNodeLabels')
                    #colorTableNode.SetNamesInitialised(True)
                    color = [0, 0, 0, 0]
                    for i,l in zip(ids,labels):
                        random_colors.GetColor(i, color)
                        color_table_node.SetColor(i, *color)
                        color_table_node.SetColorName(i, l)
                    slicer.mrmlScene.AddNode(color_table_node)

                    if len(segmentation.shape) == 3:
                        segmentation = segmentation[None,...]

                    if segmentation.shape[0] > 1:
                        seg_sequence = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', f'Seg_Seq_{node_name}')
                    for i in range(segmentation.shape[0]):
                        labelmap_node = self.create_vol_node(segmentation[i], vol_resolution, f'SEG_{node_name}', is_labelmap=True)
                        labelmap_node.GetDisplayNode().SetAndObserveColorNodeID(color_table_node.GetID())

                        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, seg_node)
                        seg_node.CreateClosedSurfaceRepresentation()
                        slicer.mrmlScene.RemoveNode(labelmap_node)

                        if seg_sequence is not None:
                            seg_sequence.SetDataNodeAtValue(seg_node, str(i))
                            slicer.mrmlScene.RemoveNode(seg_node)
                        
                            result_nodes.append(seg_sequence)

                if seg_sequence is not None or vol_sequence is not None:
                    sequence_browser = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', f'SeqBrowser_{node_name}')
                    if vol_sequence is not None: sequence_browser.AddSynchronizedSequenceNode(vol_sequence)
                    if seg_sequence is not None: sequence_browser.AddSynchronizedSequenceNode(seg_sequence)

            # show volume
            appLogic = slicer.app.applicationLogic()
            selNode = appLogic.GetSelectionNode()
            if vol_sequence is not None:
                selNode.SetReferenceActiveVolumeID(vol_sequence.GetID())
            if vol_node is not None:
                selNode.SetReferenceActiveVolumeID(vol_node.GetID())
            if proj_node is not None:
                selNode.SetReferenceActiveVolumeID(proj_node.GetID())

            appLogic.PropagateVolumeSelection()
            appLogic.FitSliceToAll()

        except Exception as e:
            import traceback
            traceback.print_exc()
            errorMessage = f"Failed to load hdf volume: {str(e)}"
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, errorMessage)
            return False
        
        return result_nodes

class LoaderWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Loader.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)


        uiWidget.setMRMLScene(slicer.mrmlScene)


        self.logic = LoaderLogic()

        # Buttons
        self.ui.loadButton.connect("clicked(bool)", self.onLoadButton)
        self.ui.inferButton.connect("clicked(bool)", self.onInferButton)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def onLoadButton(self) -> None:
        """Run processing when user clicks "Load" button."""

        self.logic.server = self.ui.server.text 
        self.logic.token = self.ui.token.text

        uri = self.ui.load_file_uri.text
        h5file = self.logic.download_file(uri, self.ui.cb_run_totalsegmentator.checked)

        LoaderFileReader(self.parent).load_from_stream(h5file, uri)

    def onInferButton(self) -> None:

        self.logic.server = self.ui.server.text 
        self.logic.token = self.ui.token.text

        run_segmentator = self.ui.cb_run_totalsegmentator.checked

        file_uri = self.ui.infer_file_uri.text
        model_uri = self.ui.infer_model_uri.text

        h5file = self.logic.run_inference(model_uri, file_uri, run_segmentator)

        reader = LoaderFileReader(self.parent)

        reader.load_from_stream(h5file, f'{model_uri}/{file_uri}')

class LoaderLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        rt = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.server_cert = os.path.join(rt, 'clt-lab-t-7200.zhaw.ch.pem') # required because we use a self-signed cert that is not trusted otherwise...
        self.token = ""
        self.server = ""

    def _get(self, path: str, params: dict = None):
        
        print(f'request {path}...')

        if params is None:
            params = {'token': self.token }
        else:
            params.update({'token': self.token })

        return requests.get(path, 
            timeout = 600, 
            verify = self.server_cert, 
            params = params)


    def getParameterNode(self):
        return LoaderParameterNode(super().getParameterNode())

    def download_file(self, uri: str, run_segmentator: bool):
        
        start = time.monotonic()

        path = f'{self.server}/image/{uri}' 

        params = {'run_totalsegmentator': run_segmentator}

        r = self._get(path, params)

        stop = time.monotonic()
        logging.info(f"Downloaded file ({int(len(r.content)/1000)}kB) completed in {stop-start:.3f} seconds")

        data = io.BytesIO(r.content)
        return data

    def run_inference(self, model_uri: str, file_uri: str, run_segmentator: bool):

        start = time.monotonic()

        path = f'{self.server}/infer/{model_uri}/{file_uri}'

        params = {'run_totalsegmentator': run_segmentator}
        r = self._get(path, params)

        stop = time.monotonic()
        logging.info(f"Downloaded file ({int(len(r.content)/1000)}kB) completed in {stop-start:.3f} seconds")

        data = io.BytesIO(r.content)
        return data


class LoaderTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_Loader1()

    def test_Loader1(self):
        pass
