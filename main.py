"""# ===========================================================================#
#                                                                           #
#  Copyright (C) 2006 - 2018                                                #
#  IDS Imaging Development Systems GmbH                                     #
#  Dimbacher Str. 6-8                                                       #
#  D-74182 Obersulm, Germany                                                #
#                                                                           #
#  The information in this document is subject to change without notice     #
#  and should not be construed as a commitment by IDS Imaging Development   #
#  Systems GmbH. IDS Imaging Development Systems GmbH does not assume any   #
#  responsibility for any errors that may appear in this document.          #
#                                                                           #
#  This document, or source code, is provided solely as an example          #
#  of how to utilize IDS software libraries in a sample application.        #
#  IDS Imaging Development Systems GmbH does not assume any responsibility  #
#  for the use or reliability of any portion of this document or the        #
#  described software.                                                      #
#                                                                           #
#  General permission to copy or modify, but not for profit, is hereby      #
#  granted, provided that the above copyright notice is included and        #
#  reference made to the fact that reproduction privileges were granted     #
#  by IDS Imaging Development Systems GmbH.                                 #
#                                                                           #
#  IDS Imaging Development Systems GmbH cannot assume any responsibility    #
#  for the use or misuse of any portion of this software for other than     #
#  its intended diagnostic purpose in calibrating and testing IDS           #
#  manufactured cameras and software.                                       #
#                                                                           #
#===========================================================================#

# Developer Note: I tried to let it as simple as possible.
# Therefore there are no functions asking for the newest driver software or freeing memory beforehand, etc.
# The sole purpose of this program is to show one of the simplest ways to interact with an IDS camera via the uEye API.
# (XS cameras are not supported)
#-------------------------------------------------------------------------------------------------------------------"""

import cv2
import numpy as np
# Libraries
from pyueye import ueye
from pyzbar.pyzbar import decode as qrDecode
from pylibdmtx.pylibdmtx import decode as dmDecode
import pytesseract
import tkinter as tk
from tkinter import messagebox

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Camera Variables -- this setup supplied by IDS for their Ueye X-series camera
hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()
pitch = ueye.INT()
nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)

op_mode = []  # dictates which decode method will be applied while running

# ------------------------------------------------------------------------------------------------------------------
# TKinter GUI structure
window = tk.Tk()
greeting = tk.Label(text="Welcome to Camera Match \n Please select which way you'd like to match.",
                    foreground="white",  # Set the text color to white
                    background="#34A2FE",  # Set the background color to light blue
                    width=125,
                    height=5).pack()


def qr_run_button():
    messagebox.showinfo("Selection Window", "QR Mode Selected!")
    op_mode.append('qrDecode()')
    window.destroy()


def dm_run_button():
    messagebox.showinfo("Selection Window", "DataMatrix Mode Selected!")
    op_mode.append('dmDecode()')
    window.destroy()


def ocr_run_button():
    messagebox.showinfo("Selection Window", "OCR Mode Selected!")
    op_mode.append('pytesseract.image_to_string(frame)')
    window.destroy()

# I need to fix the binding of the buttons. Append isn't working
btn_qrscan = tk.Button(
    master=window,
    text="Click here for QR Scanning",
    command=qr_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5).pack()
btn_dmscan = tk.Button(
    master=window,
    text="Click here for DataMatrix Scanning",
    command=dm_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5).pack()
btn_ocrscan = tk.Button(
    master=window,
    text="Click here for \n Optical Character Recognition Scanning",
    command=ocr_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5).pack()


window.mainloop()

print("START \n")
print(op_mode[0])

# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    print("is_InitCamera ERROR")

'''Reads out the data hard-coded in the non-volatile camera memory and writes it to the
 data structure that cInfo points to'''

nRet = ueye.is_GetCameraInfo(hCam, cInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetCameraInfo ERROR")

# You can query additional information about the sensor type used in the camera
nRet = ueye.is_GetSensorInfo(hCam, sInfo)
if nRet != ueye.IS_SUCCESS:
    print("is_GetSensorInfo ERROR")

nRet = ueye.is_ResetToDefault(hCam)
if nRet != ueye.IS_SUCCESS:
    print("is_ResetToDefault ERROR")

# Set display mode to DIB
nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

# Set the right color mode
if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    # setup the color depth to the current windows setting
    ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_BAYER: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
    # for color camera models use RGB32 mode
    m_nColorMode = ueye.IS_CM_BGRA8_PACKED
    nBitsPerPixel = ueye.INT(32)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_CBYCRY: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
    # for color camera models use RGB32 mode
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("IS_COLORMODE_MONOCHROME: ", )
    print("\tm_nColorMode: \t\t", m_nColorMode)
    print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
    print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
    print()

else:
    # for monochrome camera models use Y8 mode
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("else")

# Can be used to set the size and position of an "area of interest"(AOI) within an image
nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
if nRet != ueye.IS_SUCCESS:
    print("is_AOI ERROR")

width = rectAOI.s32Width
height = rectAOI.s32Height

# Prints out some information about the camera and the sensor
print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
print("Maximum image width:\t", width)
print("Maximum image height:\t", height)
print()

'''Allocates an image memory for an image having its dimensions defined by width and height
  and its color depth defined by nBitsPerPixel'''

nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    print("is_AllocImageMem ERROR")
else:
    # Makes the specified image memory the active memory
    nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_SetImageMem ERROR")
    else:
        # Set the desired color mode
        nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

# Activates the camera's live video mode (free run mode)
nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
if nRet != ueye.IS_SUCCESS:
    print("is_CaptureVideo ERROR")

# Enables the queue mode for existing image memory sequences
nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
if nRet != ueye.IS_SUCCESS:
    print("is_InquireImageMem ERROR")
else:
    print("Press q to leave the program")

# ---------------------------------------------------------------------------------------------------------------------
# create inserter and feeder lists - base has 15 positions, accumulator has 4
accumulator = []
mailBase = []
print("accumulator = ", accumulator)
print("mailBase = ", mailBase)


# --------------------------------------------------------------------------------------------------------------------
# add decoded content to the accumulator--regardless of method to decode
# if decodeContent not in accumulator:
def add_to_accumulator_queue(decoded_content):
    if len(accumulator) <= 3:
        if decoded_content not in accumulator:
            accumulator.append(decoded_content)
            print('Mailstar Base still empty; accumulator queue: ', accumulator)
    else:
        if decoded_content not in accumulator:
            accum_dump = accumulator.pop(0)
            accumulator.append(decoded_content)
            print('Accumulator Queue: ', accumulator)

            if len(mailBase) <= 14:
                mailBase.append(accum_dump)
                print('Mailstar Base Queue: ', mailBase, '\n')
            else:
                mailBase.pop(0)  # this is where the check for match occurs
                mailBase.append(accum_dump)
                print(mailBase, '\n next scan \n')


# functions for handling mis-feeds/jams

# (purge accumulator out into the mailBase -- for use if accumulator double-feeds)
def purge_accumulator():
    pass


# adding bounding box around the scanned barcode and human-readable content
def human_readability(image):
    for barcode in image:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # show what the barcode reads by extracting the byte string literal and 'decoding' to str
        decode_data = barcode.data.decode("utf-8")

        # font for overlay
        font = cv2.FONT_HERSHEY_PLAIN

        # origin of overlay
        org = (x, y + h + 70)

        # fontScale
        fontScale = 3

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        cv2.putText(frame, decode_data, org, font, fontScale, color, thickness)
        cv2.putText(frame, "Press Q to Quit", (50, 50), font, int(2), 'red', thickness)


# ---------------------------------------------------------------------------------------------------------------------

# Continuous image display
while nRet == ueye.IS_SUCCESS:

    # In order to display the image in an OpenCV window we need to extract the data of our image memory
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

    # bytes_per_pixel = int(nBitsPerPixel / 8)

    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
    image4tesseract = frame.copy()

    # ...resize the image by a half
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # ...double the image size
    # frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # ---------------------------------------------------------------------------------------------------------------------
    # Include image data processing here

    processed_image = qrDecode(frame)

    # adding bounding box around the scanned barcode and human-readable content
    for barcode in processed_image:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # show what the barcode reads by extracting the byte string literal and 'decoding' to str
        decodeQRContent = barcode.data.decode("utf-8")

        # font for overlay
        font = cv2.FONT_HERSHEY_PLAIN

        # origin of overlay
        org = (x, y + h + 70)

        # fontScale
        fontScale = 3

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        cv2.putText(frame, decodeQRContent, org, font, fontScale, color, thickness)
        cv2.putText(frame, "Press Q to Quit", (50, 50), font, int(2), color,  thickness)

        '''# while scanning for barcodes, if new barcode is detected, place it in list; when next code is scanned, move 
        all previously scanned one index position

        # Inserter and feeder index assignment -- might need to incorporate laser barcode scanner for upstream??--'''

        # if decodeContent not in accumulator:
        if len(accumulator) <= 3:
            if decodeQRContent not in accumulator:
                accumulator.append(decodeQRContent)
                print('Mailstar Base still empty; accumulator queue: ', accumulator)
        else:
            if decodeQRContent not in accumulator:
                accum_dump = accumulator.pop(0)
                accumulator.append(decodeQRContent)
                print('Accumulator Queue: ', accumulator)

                if len(mailBase) <= 14:
                    mailBase.append(accum_dump)
                    print('Mailstar Base Queue: ', mailBase, '\n')
                else:
                    mailBase.pop(0)  # this is where the check for match occurs
                    mailBase.append(accum_dump)
                    print(mailBase, '\n next scan \n')

    '''# Datamatrix decoding

    dataMatrixDecode = dmDecode(frame)
    for b in dataMatrixDecode:
        dmDecodedContent = b.data.decode("utf-8")
        if dmDecodedContent not in accumulator:
            accumulator.append(dmDecodedContent)
    print(accumulator)'''

    '''# TESSERACT DECODING -- WAAAAY SLOWER PROCESSING!  output framerate drops heavily.
    # ocr_text = pytesseract.image_to_string(frame)
    # print(ocr_text)  # anything tesseract detects gets printed!'''

    # ---------------------------------------------------------------------------------------------------------------------

    # ...and finally display it
    cv2.imshow("Press Q to Quit", frame)

    # Press q if you want to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ---------------------------------------------------------------------------------------------------------------------

# Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

# Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
ueye.is_ExitCamera(hCam)

# Destroys the OpenCv windows
cv2.destroyAllWindows()

print()
print("END")
