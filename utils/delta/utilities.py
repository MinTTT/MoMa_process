'''
This file contains functions and class definitions that are used in pipeline.py
A lot of functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

@author: jblugagne
'''
import cv2, os, re
# import numpy
import numpy as np
from joblib import Parallel, delayed
from typing import Union


def deskew(image):
    '''
    Compute rotation angle of chambers in image for rotation correction.
    From: https://gist.github.com/panzerama/beebb12a1f9f61e1a7aa8233791bc253
    Not extensively tested. You can skip rotation correction if your chambers
    are about +/- 1 degrees of horizontal

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    rotation_number : float
        Rotation angle of the chambers for correction.

    '''

    from skimage.filters import gaussian, threshold_otsu
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    # canny edges in scikit-image
    edges = canny(image)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


def cropbox(img, box):
    '''
    Crop image

    Parameters
    ----------
    img : 2D numpy array
        Image to crop.
    box : Dictionary
        Dictionary describing the box to cut out, containing the following 
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    '''
    return img[box['ytl']:box['ybr'], box['xtl']:box['xbr']]


def rangescale(frame, rescale):
    '''
    Rescale image values to be within range

    Parameters
    ----------
    frame : 2D numpy array of uint8/uint16/float/bool
        Input image.
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    '''
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame - np.min(frame)) / np.ptp(frame)) * np.ptp(rescale) + rescale[0]
    else:
        frame = np.ones_like(frame) * (rescale[0] + rescale[1]) / 2
    return frame


def driftcorr(img: np.ndarray, template=None, box=None, drift=None, parallel=True):
    '''
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drift for.
    template : None or 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        default is None.
    box : None or dictionary, optional
        A cropping box to extract the part of the frame to compute drift 
        correction over (see cropbox()).
        default is None.
    drift : None or tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        default it None.

    Returns
    -------
    2D/3D numpy array, tuple of len 2
        Drift-corrected image and drift.

    '''

    if len(img.shape) == 2:
        twoDflag = True
        img = np.expand_dims(img, axis=0)
    else:
        twoDflag = False

    if drift is None:
        if template is None:  # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0, 0)
        template = rangescale(template, (0, 255)).astype(np.uint8)  # Making sure its the right format
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]], [drift[1]])
    else:
        (xcorr, ycorr) = drift

    def parallel_drift(inx):
        if drift is None:
            frame = rangescale(img[inx], (0, 255)).astype(np.uint8)  # Making sure its the right format
            driftcorrimg = cropbox(frame, box)  # crop part of image for matching
            res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)  # top left corner
            ycorr[inx] = max_loc[0] - res.shape[1] / 2  # TODO: why size should over 2 ?
            xcorr[inx] = max_loc[1] - res.shape[0] / 2  # TODO: why size should over 2 ?
        T = np.float32([[1, 0, -ycorr[inx]],
                        [0, 1, -xcorr[inx]]])  # xcorr: direction for shiftting along y axis
        img[inx] = cv2.warpAffine(img[inx], T, img.shape[3:0:-1])

    if parallel:
        _ = Parallel(n_jobs=32, require='sharedmem')(delayed(parallel_drift)(i) for i in range(img.shape[0]))
    else:
        for i in range(img.shape[0]):
            if drift is None:
                frame = rangescale(img[i], (0, 255)).astype(np.uint8)  # Making sure its the right format
                driftcorrimg = cropbox(frame, box)  # crop part of image for matching
                res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)  # top left corner
                ycorr[i] = max_loc[0] - res.shape[1] / 2  # TODO: why size should over 2 ?
                xcorr[i] = max_loc[1] - res.shape[0] / 2  # TODO: why size should over 2 ?
            T = np.float32([[1, 0, -ycorr[i]],
                            [0, 1, -xcorr[i]]])  # xcorr: direction for shiftting along y axis
            img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])

    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    else:
        return img, (xcorr, ycorr)


def getChamberBoxes(chambersmask):
    '''
    This function extracts the bounding boxes of the chambers in thebinary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of dictionaries
        List of cropping box dictionaries (see cropbox()).

    '''
    chamberboxes = []
    if chambersmask.dtype == bool:
        chambersmask = chambersmask.astype(np.uint8)
    else:
        chambersmask = cv2.threshold(chambersmask, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(chambersmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    y_length, _ = chambersmask.shape
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamber_dict = dict(
            xtl=xtl,
            ytl=int(ytl - (boxheight * .08)),  # -10% of height to make sure the top isn't cropped
            xbr=xtl + boxwidth,
            ybr=int(ytl + boxheight * 1.1))

        if chamber_dict['ytl'] < 0:
            chamber_dict['ytl'] = 0
        if chamber_dict['ybr'] > y_length:
            chamber_dict['ybr'] = y_length

        chamberboxes.append(chamber_dict)  # tl = top left, br = bottom right. MODIFY: add more 10% of height

    chamberboxes.sort(key=lambda elem: elem['xtl'])  # Sorting by top-left X (normally sorted by Y top left)
    return chamberboxes


def getDriftTemplate(chamberboxes, img) -> Union[np.ndarray, None]:
    '''
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getChamberBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    '''

    if len(chamberboxes) == 0:
        return None
    (y_cut, x_cut) = [round(i * .01) for i in
                      img.shape]  # Cutting out 2.5% (2.5% origin) of the image on each side as drift margin
    box = dict(
        xtl=x_cut,
        xbr=-x_cut,
        ytl=y_cut,
        ybr=max(chamberboxes, key=lambda elem: elem['ytl'])['ytl'] - y_cut)
    # ybr=max(chamberboxes, key=lambda elem: elem['ytl'])['ytl'] + y_cut)
    return cropbox(img, box)


def getSinglecells(seg):
    '''
    Return masks of single cells

    Parameters
    ----------
    seg : array of uint8/uint16/float/bool
        Mask of cells. Values >0.5 will be considered cell pixels

    Returns
    -------
    singlecellseg : 3D array of uint8
        Stack of single cell masks. Each single-cell mask is stacked along the
        first axis (axis 0)

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y
    singlecellseg = np.empty([len(contours), seg.shape[0], seg.shape[1]])
    for c, contour in enumerate(contours):
        singlecellseg[c] = cv2.drawContours(np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8), [contour], 0,
                                            color=1, thickness=-1)
    return singlecellseg


def getTrackingScores(inputs, outputs):
    """
    Get overlap scores between input/target cells and tracking outputs

    Parameters
    ----------
    inputs : 2D array of floats
        Segmentation mask of input/target cells that the tracking U-Net is
        tracking against. (ie segmentation mask of the 'current'/'new' frame)
    outputs : 4D array of floats
        Tracking U-Net output. For each cell in the 'old'/'previous' frame,
        two masks were produced as potential mother and daughter cell outlines.
        (plus a third, unused mask for background)

    Returns
    -------
    scores : 3D array of floats
        Overlap scores matrix between tracking predictions and current
        segmentation mask for each new-old cell, either as a mother or a
        daughter cell. Cells from the 'old' frame (axis 0) will get a tracking
        score that corresponds to how much the U-Net tracking output mask
        overlaps with the segmentation mask of each cell in the 'new' mask
        (axis 1). Each old cell gets a tracking scores for each new cell as
        either a potential daughter (mother-daughter relationship, index 1 on
        axis 2) or as just itself (mother-mother relationship, index 0 on axis
        2)
        Example:
            attrib[1,3,0] == 1 means the 'mother' tracking output of cell #2 in
            the old frame completely overlaps with the mask of cell #4 in the
            new frame.
            attrib[4,6,1] == 0.7 means the 'daughter' tracking output of cell
            #5 in the old frame overlaps with 70% of the mask of cell #7 in the
            new frame.

    """
    targetcells = getSinglecells(inputs).astype(np.uint8)
    scores = np.zeros([outputs.shape[0], targetcells.shape[0], 2], dtype=np.float32)
    for o in range(outputs.shape[0]):
        mother_pred = cv2.threshold(outputs[o, :, :, 0], .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        daughter_pred = cv2.threshold(outputs[o, :, :, 1], .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        for i in range(targetcells.shape[0]):
            scores[o, i, 0] = getOverlap(mother_pred, targetcells[i, :, :])
            scores[o, i, 1] = getOverlap(daughter_pred, targetcells[i, :, :])
    return scores


def getOverlap(output, target):
    """
    Get portion of tracking output overlapping on target cell

    Parameters
    ----------
    output : array of uint8
        Mask of tracking output.
    target : array of uint8
        Mask of target cell.

    Returns
    -------
    float
        Portion of target cell that is covered by the tracking output.

    """
    return cv2.sumElems(
        cv2.bitwise_and(output,
                        target)
    )[0] / \
           cv2.sumElems(
               target
           )[0]


def getAttributions(scores):
    '''
    Get attribution matrix from tracking scores

    Parameters
    ----------
    scores : numpy 3D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : numpy 3D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each new cell can be attributed to an
        old cell as a "mother", ie the same cell (index 0 along axis 2), or as
        a "daughter" of that old cell (index 1 along axis 2)
        Example:
            attrib[1,3,0] == True means cell #4 in the new frame (from top to 
            bottom) is attributed to cell #2 in the old frame.
            attrib[4,6,1] == True means cell #7 in the new frame is 
            attributed as a daughter to cell #5 in the old frame.
        Each 'old' cell can have only 0 or 1 'new' cell identified as itself 
        (ie mother-mother relationship), and 0 or 1 'new' cell identified as a
        daughter (ie mother-daughter relationship). Each 'new' cell can only be
        attributed to 0 or 1 'old' cell, either as a 'mother' or 'daughter'.

    '''
    attrib = np.zeros(scores.shape)
    for i in range(scores.shape[1]):  # Go over cells in the *new* frame
        if np.sum(scores[:, i, :] > 0) == 1:  # If only one old cell was attributed to the new cell
            attrib[:, i, :] = scores[:, i, :] > 0
        elif np.sum(scores[:, i, :] > 0) > 1:  # If conflicts
            if np.sum(scores == np.max(scores[:, i, :])) == 1:  # One cell has a higher score than the others
                attrib[:, i, :] = scores[:, i, :] == np.max(scores[:, i, :])
            elif np.any(scores[:, i, 1] == np.max(scores[:, i,
                                                  :])):  # At least one of the highest scoring old cells would be the mother of this new "daughter"
                attrib[np.argmax(scores[:, i,
                                 1]), i, 1] = True  # keep only the first one (which is also the one higher in the image)
            else:  # If only mother-to-mother couplings are in conflict
                attrib[np.argmax(scores[:, i,
                                 0]), i, 0] = True  # keep only the first one (which is also the one higher in the image)
    for o in range(scores.shape[0]):  # Go over cells in the *old* frame
        if np.sum(attrib[o, :,
                  0] > 0) > 1:  # If one old cell gets attributed to more than one new cell (mother-mother coupling)
            tokeep = np.argmax(attrib[o, :, 0])
            attrib[o, :, 0] = False
            attrib[o, tokeep, 0] = True  # keep only the first one
        if np.sum(attrib[o, :,
                  1] > 0) > 1:  # If one old cell gets attributed to more than one new cell (mother-daughter coupling)
            tokeep = np.argmax(attrib[o, :, 1])
            attrib[o, :, 1] = False
            attrib[o, tokeep, 1] = True  # keep only the first one
        if np.any(attrib[o, :, 0]) and np.any(attrib[o, :, 1]) and np.argmax(attrib[o, :, 0]) > np.argmax(
                attrib[o, :, 1]):  # If somehow the mother is lower than the daughter
            attrib[o, np.argmax(attrib[o, :, 0]), 1] = True
            attrib[o, np.argmax(attrib[o, :, 0]), 0] = False
            attrib[o, np.argmax(attrib[o, :, 1]), 0] = True
            attrib[o, np.argmax(attrib[o, :, 1]), 1] = False
    return attrib


def label_seg(seg, cellnumbers=None):
    '''
    Label cells in segmentation mask

    Parameters
    ----------
    seg : numpy 2D array of float/uint8/uint16/bool
        Cells segmentation mask. Values >0.5 will be considered cell pixels
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is 
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    label : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels 
        with values given by cellnumbers

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg, .5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y
    label = np.zeros(seg.shape, dtype=np.uint16)  # Because we use uint16, we can not have >65535 cells in one movie
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(label, [contour], c + 1 if cellnumbers is None else cellnumbers[c])
    return label


def updatelineage(newframe, stack, framenb=0, lineage=None, attrib=None):
    '''
    Update lineage list and labels stack based on attribution matrix

    Parameters
    ----------
    newframe : numpy 2D array
        Segmentation output for current frame.
    stack : numpy 3D tensor of uint16
        Labelled stack. Each frame until the current frame contains a numbered 
        mask, with the number corresponding the cell number in the lineage. See
        label_seg()
    framenb : int, optional
        Current frame number (0-based indexing).
        The default is 0.
    lineage : List of dicts, optional
        List of all the cells encountered in the chamber. Each cell dict 
        contains the following keys:
            - mother, int: The number of the mother cell in the lineage 
            (1-based indexing)
            - framenbs, list of ints: The list of frames where the cell is 
            present (1-based indexing)
            - daughters, list of ints: A list of the same length as framenbs,
            with each element equal to the number of the daughter in the 
            lineage (1-based indexing) if a division occured at the 
            corresponding frame number, and equal to 0 otherwise.
        Note: We used 1-based indexing for compatibility with earlier code that
        was written in Matlab.
        If None is provided, the lineage will be initialized and all cells in 
        the newframe image will be given 0 as a mother.
        The default is None.
    attrib : numpy 2D array of bool, optional
        Attribution matrix for the current frame/timepoint as produced by 
        number 0. If None, lineage and stack will not be updated.
        getAttributions(). New cells with no attributions are given mother cell
        The default is None.

    Returns
    -------
    lineage : List of dicts
        lineage updated for the current frame and attrib matrix. See lineage 
        input description above.
    stack : numpy 3D tensor of uint16
        labelled stack updated for the current frame and attrib matrix. See 
        lineage input description above..

    '''

    if lineage == None:  # Not initialized yet
        lineage = []
        stack[framenb] = label_seg(newframe)
        cells = np.unique(stack[framenb])
        for c in cells[1:]:  # Avoiding cell==0 (background)
            lineage.append(createcell(framenb + 1))  # We're using framenb +1 to be compatible with old Matlab code
    else:  # Update according to attribution matrix
        # Init cell numbers, retrieve previous stack frame:
        cellnumbers = []
        label_old = label_seg((stack[framenb - 1] > 0).astype(np.uint8))

        # If previous stack frame has no cells:
        if not np.any(
                label_old > 0):  # This can happen if the chamber is empty and suddenly cells appear (can be a segmentation error, or a focus problem)
            contours = cv2.findContours(newframe.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                0]  # In this case just label
            for _ in range(len(contours)):
                lineage.append(createcell(framenb))
                cellnumbers.append(len(lineage))

        # If previous stack frame has cells:
        else:
            for i in range(attrib.shape[1]):
                if np.any(attrib[:, i, 0]):  # If cell is tracked from an old cell: (mother-mother)
                    cellnum = int(stack[framenb - 1][label_old == (np.nonzero(attrib[:, i, 0])[0] + 1)[0]][
                                      0])  # retrieving cell number from stack
                    lineage[cellnum - 1]['framenbs'].append(float(framenb + 1))  # Update frame number list
                    if len(lineage[cellnum - 1]['daughters']) < len(lineage[cellnum - 1][
                                                                        'framenbs']):  # If framenbs hasn't been updated (in case somehow daughter goes through before mother)
                        lineage[cellnum - 1]['daughters'].append(0.)
                    cellnumbers.append(cellnum)
                elif np.any(attrib[:, i, 1]):  # Cell is tracked as a daughter cell (mother-daughter)
                    mothernb = int(stack[framenb - 1][label_old == (np.nonzero(attrib[:, i, 1])[0] + 1)[0]][
                                       0])  # retrieving cell number from stack
                    # Create new cell, as daughter of mother:
                    lineage.append(createcell(framenb + 1, mothernb))
                    cellnumbers.append(len(lineage))
                    # Update daughters of mother cell:
                    if len(lineage[mothernb - 1]['daughters']) < len(lineage[mothernb - 1]['framenbs']):
                        lineage[mothernb - 1]['daughters'].append(float(cellnumbers[-1]))
                    else:
                        lineage[mothernb - 1]['daughters'][-1] = float(cellnumbers[-1])
                else:  # Orphan cell
                    lineage.append(createcell(framenb))
                    cellnumbers.append(len(lineage))

        # Compile new labeled frame, push to stack:
        stack[framenb] = label_seg(newframe, cellnumbers=cellnumbers)
    return lineage, stack


def createcell(framenb, mothernb=0):
    '''
    Create cell to append to lineage list

    Parameters
    ----------
    framenb : int
        Frame that the cell first appears (1-based indexing).
    mothernb : int, optional
        Number of the mother cell in the lineage (1-based indexing).
        The default is 0. (ie unknown mother)

    Returns
    -------
    dict
        Initialized cell dictionary.

    '''
    return dict(framenbs=[float(framenb)],  # For compatibility with old version of the pipeline
                mothernb=mothernb,
                daughters=[0.])


class xpreader:

    def __init__(self, filename, channelnames=None, use_bioformats=False, prototype=None, fileorder='pct',
                 filenamesindexing=1):
        '''
        Initialize experiment reader

        Parameters
        ----------
        filename : String
            Path to experiment tif file or directory.If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If no prototype is provided, the filenames 
            are expected to be of the following C-style format: 
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded 
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
                Pos01_Cha3_Fra0005.tif
                p3c1t034.tif
                xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            DEPRECATED - bioformats does not work with python 3
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager 
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same 
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing = int
            Selects between 0-based or 1-based indexing in the filename. If 
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1
            

        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the 
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        '''

        _, file_extension = os.path.splitext(filename)
        self.use_bioformats = use_bioformats
        self.channelnames = channelnames
        if use_bioformats:  # Unfortunately bioformats only works with python 2, leaving it here if it helps anybody.
            import bioformats, javabridge
            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(filename)
            md = bioformats.OMEXML(bioformats.get_omexml_metadata(path=filename))
            self.positions = md.get_image_count()
            self.timepoints = md.image(0).Pixels.SizeT  # Here I'm going to assume all series have the same format
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=False, c=0).dtype

        elif os.path.isdir(filename):  # Experiment is stored as individual image TIFF files in a folder
            self.filetype = 'dir'
            self.filehandle = filename
            self.fileorder = fileorder
            self.filenamesindexing = filenamesindexing
            # If filename prototype is not provided, guess it from the first file:
            if prototype is None:
                imgfiles = [x for x in os.listdir(filename) if os.path.splitext(x)[1].lower() in ('.tif', '.tiff')]
                # Here we assume all images in the folder follow the same naming convention:
                numstrs = re.findall("\d+", imgfiles[0])  # Get digits sequences in first filename
                charstrs = re.findall("\D+", imgfiles[0])  # Get character sequences in first filename
                if len(numstrs) != 3 or len(charstrs) != 4:
                    raise ValueError('Filename formatting error. See documentation for image sequence formatting')
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype = charstrs[0] + '%0' + str(len(numstrs[0])) + 'd' + \
                                 charstrs[1] + '%0' + str(len(numstrs[1])) + 'd' + \
                                 charstrs[2] + '%0' + str(len(numstrs[2])) + 'd' + \
                                 charstrs[3]
            else:
                self.prototype = prototype
            # Get experiment settings by testing if relevant files exist: 
            # Get number of positions:
            self.positions = 0
            while (os.path.exists(self.getfilenamefromprototype(self.positions, 0, 0))): self.positions += 1
            # Get number of channels:
            self.channels = 0
            while (os.path.exists(self.getfilenamefromprototype(0, self.channels, 0))): self.channels += 1
            # Get number of frames/timepoints:
            self.timepoints = 0
            while (os.path.exists(self.getfilenamefromprototype(0, 0, self.timepoints))): self.timepoints += 1
            # Get image specs:
            I = cv2.imread(self.getfilenamefromprototype(0, 0, 0), cv2.IMREAD_ANYDEPTH)
            self.x = I.shape[1]
            self.y = I.shape[0]
            self.dtype = I.dtype

        elif (
                file_extension.lower() == '.tif' or file_extension.lower() == '.tiff'):  # Works with single-series tif & mutli-series ome.tif
            from skimage.external.tifffile import TiffFile
            self.filetype = 'tif'
            self.filehandle = TiffFile(filename)
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[0]  # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find('T')]
            self.channels = s.shape[s.axes.find('C')]
            self.x = s.shape[s.axes.find('X')]
            self.y = s.shape[s.axes.find('Y')]
            self.dtype = s.pages[0].asarray().dtype

    def close(self):
        if self.use_bioformats:
            self.filehandle.close()
            import javabridge
            javabridge.kill_vm()
        elif self.filetype == 'tif':  # Nothing to do if sequence directory
            self.filehandle.close()

    def getfilenamefromprototype(self, position, channel, frame):
        '''
        Generate full filename for specific frame based on file path, 
        prototype, fileorder, and filenamesindexing

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        '''
        filenumbers = []

        for i in self.fileorder:
            if i == 'p':
                filenumbers.append(position + self.filenamesindexing)
            if i == 'c':
                filenumbers.append(channel + self.filenamesindexing)
            if i == 't':
                filenumbers.append(frame + self.filenamesindexing)
        return os.path.join(self.filehandle, self.prototype % tuple(filenumbers))

    def getframes(self, positions=None, channels=None, frames=None,
                  squeeze_dimensions=True,
                  resize=None,
                  rescale=None,
                  rotate=None):
        '''
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer 
            or a tuple/list will be returned. If None is passed, all positions 
            are returned. 
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer 
            or a tuple/list will be returned. If the channel names have been 
            defined, the channel(s) can be passed as a string or tuple/list of 
            strings. If None is passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list 
            will be returned. If None is passed, all frames are returned. 
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        '''

        # Handle options:
        if frames is None:  # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            frames = [frames]

        if positions is None:  # No positions specified: load all
            positions = list(range(self.positions))
        elif type(positions) is not list and type(positions) is not tuple:
            positions = [positions]

        if channels is None:  # No channel specified: load all
            channels = list(range(self.channels))
        elif type(channels) is str:  # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels + ' is not a valid channel name.')
        elif (type(channels) is list or type(channels) is tuple) and all(
                [type(c) is str for c in channels]):  # refer to channels by list/tuple of names:
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            for i, c in enumerate(channels):
                if c in self.channelnames:
                    channels[i] = self.channelnames.index(c)
                else:
                    raise ValueError(c + ' is not a valid channel name.')
        elif type(channels) is not list and type(channels) is not tuple:
            channels = [channels]

        # Allocate memory:
        if rescale is None:
            dt = self.dtype
        else:
            dt = np.float32
        if resize is None:
            output = np.empty([len(positions), len(frames), len(channels), self.y, self.x], dtype=dt)
        else:
            output = np.empty([len(positions), len(frames), len(channels), resize[0], resize[1]], dtype=dt)

        # Load images:
        for p, pos in enumerate(positions):
            for c, cha in enumerate(channels):
                for f, fra in enumerate(frames):
                    # Read frame:
                    if self.use_bioformats:
                        frame = self.filehandle.read(series=pos, c=cha, t=fra, rescale=False)
                    elif self.filetype == 'dir':
                        frame = cv2.imread(self.getfilenamefromprototype(pos, cha, fra), cv2.IMREAD_ANYDEPTH)
                    elif self.filetype == 'tif':
                        frame = self.filehandle.series[pos]. \
                            pages[fra * self.channels + cha]. \
                            asarray()
                    # Optionally resize and rescale:
                    if rotate is not None:
                        M = cv2.getRotationMatrix2D((self.x / 2, self.y / 2), rotate, 1)
                        frame = cv2.warpAffine(frame, M, (self.x, self.y), borderMode=cv2.BORDER_REPLICATE)
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1])  # cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame, rescale)
                    # Add to output array:
                    output[p, f, c, :, :] = frame

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output
