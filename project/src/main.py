'''
Created on 30.10.2012

@author: David
'''
import getopt
import sys
import calibrationwrapper
import os


def usage():
    print("Flags:")
    print("\t-c <calibrationimage>")
    print("\t\tstarts the calibration process:")
    print("\t\t\t*first the program finds corners via the harris corner detector")
    print("\t\t\t*then you have to click the corners of each face to define it (see report/howto.png for the correct order of the click sequence)")
    print("\t\t\t*after each definition of a face (4 clicks) the correct coordinates of the corners should pop up")
    print("\t\t\t*after the definition of the 3 faces, the calibration process should start and you should see the results in the console and the reprojected corners in the image")
    

def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "c:")
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == "-c":
            # test all images one after the other
            # for files in os.listdir("../../data/calibration"):
            #   calibrationwrapper.calibrate("../../data/calibration/" + files)
            calibrationwrapper.calibrate(a)
            
if __name__ == '__main__':
    main()
