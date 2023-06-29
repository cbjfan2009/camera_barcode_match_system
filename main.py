# Libraries
import cv2
import numpy as np
from pyueye import ueye
import zxingcpp
import pytesseract
import tkinter as tk
from tkinter import messagebox
import imutils
#from .usps_barcode_decoder import usps_barcode_decoder

# Installed location of Tesseract-OCR in your system
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
greeting = tk.Label(text="Welcome to Camera Match. \n Please select which decoding method to use.",
                    foreground="white",  # Set the text color to white
                    background="#34A2FE",  # Set the background color to light blue
                    width=50,
                    height=5,
                    font=("Arial", 25)).pack()

def calibrate_button():
    messagebox.showinfo("Selection Window", "Setup Mode Selected!")
    op_mode.append('calibrate')
    window.destroy()

def barcode_run_button():
    messagebox.showinfo("Selection Window", "Barcode Mode Selected!")
    op_mode.append('barcodeDecode')
    window.destroy()

def usps_run_button():
    messagebox.showinfo("Selection Window", "USPS IM barcode Mode Selected!")
    op_mode.append('uspsDecode')
    window.destroy()

def ocr_run_button():
    messagebox.showinfo("Selection Window", "OCR Mode Selected!")
    op_mode.append('ocrDecode')
    window.destroy()

btn_calibrate = tk.Button(
    master=window,
    text="Camera Setup Mode",
    command=calibrate_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5,
    font=("Arial", 15)).pack()
btn_barcodescan = tk.Button(
    master=window,
    text="Click here for barcode scanning \n UNAVAILABLE WITHOUT \n WIDE-ANGLE LENS",
    command=barcode_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5,
    font=("Arial", 15)).pack()
btn_uspsscan = tk.Button(
    master=window,
    text="Click here for USPS IMb scanning",
    command=usps_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5,
    font=("Arial", 15)).pack()
btn_ocrscan = tk.Button(
    master=window,
    text="Click here for \n Optical Character Recognition scanning \n Warning: VERY SLOW!",
    command=ocr_run_button,
    width=35,
    height=5,
    bg="blue",
    fg="yellow",
    relief=tk.RAISED,
    borderwidth=5,
    font=("Arial", 15)).pack()


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

'''define the length of time: time_exposure_,
    make it into a C API accessible double,
    call the ueye command (IS_EXPOSURE...) using the double, and tell the camera
    how many bites the double is (sizeof...) --requirement of c?
    '''
time_exposure_ = 4  # I think this is in milliseconds?

time_exposure = ueye.double(time_exposure_)
nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))

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


# create inserter and feeder lists - base has 15 positions, accumulator has 4 -----------------------------------------
accumulator = []
mailBase = []
print("accumulator = ", accumulator)
print("mailBase = ", mailBase)


# Continuous image display---------------------------------------------------------------------------------------------
while nRet == ueye.IS_SUCCESS:

    # In order to display the image in an OpenCV window we need to extract the data of our image memory
    array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

    # bytes_per_pixel = int(nBitsPerPixel / 8)

    # ...reshape it in an numpy array...
    frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))

    # ...double the image size
    # frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)       # FRAME IS 3-channel UINT8 format


    # Include image data processing here--------------------------------------------------------------------------------
    if op_mode[0] == 'barcodeDecode':
        results = zxingcpp.read_barcodes(frame)
        for r in results:
            #print useful data to console for troubleshooting
            #print(f"Text:          '{r.text}'")
            #print(f"Symbology:     {r.format.name}")
            #print(f"Content Type:  {r.content_type.name}")
            #print(f"Bounding Box:  {r.position}")
            #print(f"Rotation:      {r.orientation}deg")
            #print()

            # when converting r.position to a string, there is a NUL character appended, which must be removed
            t = str(r.position)
            t = t.replace("\x00", "")
            t = [list(map(int, x.split("x"))) for x in t.split(" ")]


            coords = {
                "top_right": {
                    "x": t[0][0],
                    "y": t[0][1],
                },
                "bottom_right": {
                    "x": t[1][0],
                    "y": t[1][1],
                },
                "bottom_left": {
                    "x": t[2][0],
                    "y": t[2][1],
                },
                "top_left": {
                    "x": t[3][0],
                    "y": t[3][1],
                },
            }

            cv2.rectangle(
                frame,
                (
                    min(coords["top_right"]["x"], coords["top_left"]["x"]),
                    min(coords["top_right"]["y"], coords["top_left"]["y"]),
                ),
                (
                    max(coords["bottom_right"]["x"], coords["bottom_left"]["x"]),
                    max(coords["bottom_right"]["y"], coords["bottom_left"]["y"]),
                ),
                (0, 0, 255),
                2,
            )
            # if decodeContent not in accumulator:
            if len(accumulator) <= 3:
                if r.text not in accumulator:
                    accumulator.append(r.text)
                    print('Mailstar Base still empty; accumulator queue: ', accumulator)
            else:
                if r.text not in accumulator:
                    accum_dump = accumulator.pop(0)
                    accumulator.append(r.text)
                    print('Accumulator Queue: ', accumulator)

                    if len(mailBase) <= 14:
                        mailBase.append(accum_dump)
                        print('Mailstar Base Queue: ', mailBase, '\n')
                    else:
                        mailBase.pop(0)  # this is where the check for match occurs
                        mailBase.append(accum_dump)
                        print(mailBase, '\n next scan \n')

        if len(results) == 0:
            print("Could not find any barcode.")


    elif op_mode[0] == 'uspsDecode':
        try:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # binary image processing-----------------------------------------------------------------------------------
            ret, thresh = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY_INV)
            thresh_copy = thresh.copy()

            # visualize the binary image
            #cv2.imshow('Threshold', thresh)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # locating the largest bounding box to select the AOI (barcode region)--------------------------------------
            ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
            gradX = cv2.Sobel(thresh, ddepth=ddepth, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(thresh, ddepth=ddepth, dx=0, dy=1, ksize=-1)

            # subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)
            blurred = cv2.blur(gradient, (1, 1))
            (_, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            #cv2.imshow("thresh", thresh)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # construct a closing kernel and apply it to the thresholded image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # perform a series of erosions and dilations
            closed = cv2.erode(closed, None, iterations=4)
            closed = cv2.dilate(closed, None, iterations=4)

            # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
            cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]


            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
            box = np.intp(box)

            # find (x,y) coordinate for line dissecting the bounding box
            x_coord = []
            y_coord = []
            for points in box:
                x_coord.append(points[0])
                y_coord.append(points[1])
            y_min = min(y_coord)
            y_max = max(y_coord)
            y_mid = (y_max + y_min) // 2
            x_max = max(x_coord)
            x_min = min(x_coord)

            # use barcode bounding box ROI as area to search with connectedcomponentswithstats()
            roi_image_copy = thresh_copy[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi_image_copy, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # will try turning barcode matrix sideways to see if connectedcomponentswithstats() will iterate better??
            roi_image_copy = cv2.rotate(roi_image_copy, cv2.ROTATE_90_CLOCKWISE)  # if I need to run it sideways to iterate.....?
            height, width = thresh.shape[:2]

            # cv.imshow("Barcode ROI for contours analysis", roi_resized) # shows rescaled version of the ROI

            xR, yR, wR, hR = cv2.boundingRect(roi_image_copy)
            # print(xR, yR, wR, hR)

            # connected components approach--------------------------------------------------------------------------------
            analysis = cv2.connectedComponentsWithStats(roi_image_copy, 4, cv2.CV_32S)  # integer 4 is the connectivity 4 vs 8 way
            (totalLabels, label_ids, values, centroid) = analysis
            output = np.zeros(roi_image_copy.shape, dtype="uint8")

            # attempt to sort by centroid X-axis locations, so I can work left-to-right on barcode.
            #y_coords = centroid[:, 1]  # use all of them. background skipped later.
            #indices = np.argsort(y_coords)

            barcode_translation = []
            counter = 1
            # Loop through each component
            for i in range(1, totalLabels):
                # Area of the component
                area = values[i, cv2.CC_STAT_AREA]

                if (area > 1) and (area < 400):
                    # Create a new image for bounding boxes
                    new_img = roi_image_copy.copy()

                    # Now extract the coordinate points
                    x1 = values[i, cv2.CC_STAT_LEFT]
                    y1 = values[i, cv2.CC_STAT_TOP]
                    w = values[i, cv2.CC_STAT_WIDTH]
                    h = values[i, cv2.CC_STAT_HEIGHT]

                    # Coordinate of the bounding box for each bar
                    pt1 = (x1, y1)
                    pt2 = (x1 + w, y1 + h)
                    (X, Y) = centroid[i]

                    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

                    # Bounding boxes for each component
                    # cv.rectangle(new_img, pt1, pt2, (255, 255, 0), 3)
                    # cv.circle(new_img, (int(X),int(Y)), 4, (0, 255, 255), -1)

                    # Create a new array to show individual component
                    component = np.zeros(roi_image_copy.shape, dtype="uint8")
                    componentMask = (label_ids == i).astype("uint8") * 255

                    # Apply the mask using the bitwise operator
                    component = cv2.bitwise_or(component, componentMask)
                    output = cv2.bitwise_or(output, componentMask)

                    # print to console the x-min/x-max (yields height of bar and offset from tracking)
                    # print("Bar # ", counter)
                    # print("X-min = ", x1, "X=max = ", x1+w)
                    counter += 1

                    '''
                    barcode bars are F, A, D, T
                    F = x_min to x_max
                    A = tracking min_x to x_max
                    D = x_min to tracking max_x
                    T = neither max_x or min_x
                    '''
                    if x1 == xR and (x1 + w) == (
                            xR + wR):  # full bar - touches x=0 and touches highest x-value (for sample image==24)
                        barcode_translation.append('F')
                    elif x1 == xR and (x1 + w) != (xR + wR):  # descending bar - touches x=0, doesn't touch highest x-value
                        barcode_translation.append('D')
                    elif x1 != xR and (x1 + w) == (
                            xR + wR):  # ascending bar - doesn't touch x=0, does touch highest x-value
                        barcode_translation.append('A')
                    else:
                        barcode_translation.append('T')  # doesn't touch x=0 or high-x so it's the short bar for tracking

                    # Show the final images
                    # cv.imshow("Image", new_img)
                    # cv.imshow("Individual Component", component)
                    # cv.imshow("Filtered Components", output)
                    # cv.waitKey(0)

            print(barcode_translation)
            if len(accumulator) <= 3:
                if barcode_translation not in accumulator:
                    accumulator.append(barcode_translation)
                    print('Mailstar Base still empty; accumulator queue: ', accumulator)
            else:
                if barcode_translation not in accumulator:
                    accum_dump = accumulator.pop(0)
                    accumulator.append(barcode_translation)
                    print('Accumulator Queue: ', accumulator)

                    if len(mailBase) <= 14:
                        mailBase.append(accum_dump)
                        print('Mailstar Base Queue: ', mailBase, '\n')
                    else:
                        mailBase.pop(0)  # this is where the check for match occurs
                        mailBase.append(accum_dump)
                        print(mailBase, '\n next scan \n')
        except:
            cv2.imshow("Frame", frame)

    elif op_mode[0] == 'ocrDecode':
        # TESSERACT DECODING -- SIGNIFICANTLY SLOWER PROCESSING!  Output framerate drops heavily -- needs GPU acceleration
        ocr_text = pytesseract.image_to_string(frame, lang='eng')
        print(ocr_text)  # anything tesseract detects gets printed -- needs to be implemented with a sensor/trigger!


    elif op_mode[0] == 'calibrate':
        results = zxingcpp.read_barcodes(frame)
        overlay_frame = frame.copy()
        datamatrix_sample = cv2.imread('C:\\Users\\prodsupervisor\\Desktop\\datamatrix_samples.jpg')


        cv2.imshow("If using Data Matrix, consider this: ", datamatrix_sample)
        cv2.line(frame, (0, 170), (640, 170), (255,255,255), 2)  #add mode lines to help target scanner
        cv2.line(frame, (0, 310), (640, 310), (255, 255, 255), 2)
        cv2.imshow("Barcode Targeting", frame)
        for r in results:
            t = str(r.position)
            t = t.replace("\x00", "")
            t = [list(map(int, x.split("x"))) for x in t.split(" ")]

            coords = {
                "top_right": {
                    "x": t[0][0],
                    "y": t[0][1],
                },
                "bottom_right": {
                    "x": t[1][0],
                    "y": t[1][1],
                },
                "bottom_left": {
                    "x": t[2][0],
                    "y": t[2][1],
                },
                "top_left": {
                    "x": t[3][0],
                    "y": t[3][1],
                },
            }

            cv2.rectangle(
                frame,
                (
                    min(coords["top_right"]["x"], coords["top_left"]["x"]),
                    min(coords["top_right"]["y"], coords["top_left"]["y"]),
                ),
                (
                    max(coords["bottom_right"]["x"], coords["bottom_left"]["x"]),
                    max(coords["bottom_right"]["y"], coords["bottom_left"]["y"]),
                ),
                (0, 0, 255),
                2,
            )

            # font for overlay
            font = cv2.FONT_HERSHEY_PLAIN

            # origin of overlay
            org = (50, 50)

            # fontScale
            fontScale = 3

            # Red color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            cv2.putText(frame, "Decode Successful", org, font, fontScale, color, thickness)


# ...and finally display it
    cv2.imshow("Press Q to Quit", frame)

    # Press q if you want to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break